import os
import time
import json
import logging
from typing import Tuple, List, Optional

import requests
import gspread

# -----------------------------------
# Logging setup
# -----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------------
# Config / Environment
# -----------------------------------
FOREX_NEWS_API_KEY = os.getenv("FOREX_NEWS_API_KEY")

GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Active-Investing")
GOOGLE_SHEET_TAB = os.getenv("GOOGLE_SHEET_TAB", "Oanda-Screener")

UPDATE_INTERVAL_SECONDS = int(os.getenv("FOREX_NEWS_INTERVAL_SECONDS", "21600"))

# We will ONLY write to T and U:
# T = news_top_keywords_7d
# U = news_emoji_7d
NEWS_HEADER_RANGE = "T1:U1"
NEWS_DATA_START_ROW = 2  # data begins on row 2
NEWS_DATA_COLUMNS_RANGE_TEMPLATE = "T{start_row}:U{end_row}"

NEWS_TOP_KEYWORDS_COL_HEADER = "news_top_keywords_7d"
NEWS_EMOJI_COL_HEADER = "news_emoji_7d"

FOREX_NEWS_STAT_URL = "https://forexnewsapi.com/api/v1/stat"

# -----------------------------------
# Universe of pairs (68 total)
# (Matches the order seen in your logs)
# -----------------------------------
FOREX_PAIRS: List[str] = [
    "NZD_SGD",
    "USD_SGD",
    "EUR_SEK",
    "GBP_NZD",
    "EUR_PLN",
    "AUD_CAD",
    "GBP_CAD",
    "USD_MXN",
    "GBP_USD",
    "AUD_USD",
    "GBP_PLN",
    "USD_TRY",
    "GBP_JPY",
    "SGD_CHF",
    "SGD_JPY",
    "GBP_ZAR",
    "USD_JPY",
    "EUR_TRY",
    "EUR_JPY",
    "AUD_SGD",
    "EUR_NZD",
    "GBP_HKD",
    "CHF_JPY",
    "EUR_HKD",
    "USD_THB",
    "GBP_CHF",
    "AUD_CHF",
    "NZD_CHF",
    "AUD_HKD",
    "USD_CHF",
    "CAD_HKD",
    "USD_HKD",
    "NZD_JPY",
    "ZAR_JPY",
    "AUD_JPY",
    "EUR_SGD",
    "TRY_JPY",
    "CHF_HKD",
    "GBP_SGD",
    "USD_SEK",
    "NZD_HKD",
    "USD_CNH",
    "USD_CZK",
    "EUR_GBP",
    "EUR_NOK",
    "USD_CAD",
    "EUR_AUD",
    "CAD_CHF",
    "AUD_NZD",
    "HKD_JPY",
    "USD_NOK",
    "GBP_AUD",
    "USD_PLN",
    "EUR_ZAR",
    "NZD_USD",
    "USD_ZAR",
    "CAD_JPY",
    "CAD_SGD",
    "USD_HUF",
    "EUR_CAD",
    "CHF_ZAR",
    "USD_DKK",
    "EUR_HUF",
    "EUR_CHF",
    "EUR_DKK",
    "EUR_USD",
    "EUR_CZK",
    "NZD_CAD",
]


# -----------------------------------
# Google Sheets helpers
# -----------------------------------
def get_gspread_client() -> gspread.Client:
    if not GOOGLE_CREDS_JSON:
        raise RuntimeError("GOOGLE_CREDS_JSON is not set")

    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    client = gspread.service_account_from_dict(creds_dict)
    return client


def get_worksheet() -> gspread.Worksheet:
    client = get_gspread_client()
    sh = client.open(GOOGLE_SHEET_NAME)
    ws = sh.worksheet(GOOGLE_SHEET_TAB)
    return ws


# -----------------------------------
# ForexNewsAPI /stat helpers
# -----------------------------------
def pair_to_api_symbol(pair: str) -> str:
    """
    Convert our underscore pair (e.g. 'EUR_USD') to API symbol (e.g. 'EUR-USD').
    """
    return pair.replace("_", "-")


def classify_emoji_from_stat(stat: dict) -> str:
    """
    Map sentiment stats to an emoji.

    This is intentionally simple & robust — if the fields are missing, fall back
    to neutral / no-info icons without throwing.
    """
    if not isinstance(stat, dict):
        return "➖"

    sentiment = stat.get("sentiment") or {}
    try:
        bullish = float(sentiment.get("bullish", 0) or 0)
        bearish = float(sentiment.get("bearish", 0) or 0)
    except (TypeError, ValueError):
        bullish, bearish = 0.0, 0.0

    total = bullish + bearish

    # No news or not enough data
    if total < 0.1:
        return "➖"

    # Strongly bullish
    if bullish > 0.55 and bullish - bearish > 0.15:
        return "🟢"

    # Strongly bearish
    if bearish > 0.55 and bearish - bullish > 0.15:
        return "🔴"

    # Otherwise "mixed / neutral"
    return "⚪"


def extract_top_keywords(stat: dict, limit: int = 5) -> str:
    """
    Extract top keywords from the stat block and join them into a string.

    We try to be robust against whatever format comes back (list of strings,
    list of dicts, etc).
    """
    if not isinstance(stat, dict):
        return ""

    raw_kw = stat.get("top_keywords", []) or stat.get("keywords", []) or []

    keywords: List[str] = []

    if isinstance(raw_kw, list):
        for item in raw_kw:
            if isinstance(item, str):
                keywords.append(item)
            elif isinstance(item, dict):
                # Common pattern: {"keyword": "...", "count": ...}
                val = item.get("keyword")
                if isinstance(val, str):
                    keywords.append(val)
    elif isinstance(raw_kw, dict):
        # If it's a dict, maybe {"keyword1": count, "keyword2": count}
        for key in raw_kw.keys():
            if isinstance(key, str):
                keywords.append(key)

    # Deduplicate and limit
    cleaned = [k.strip() for k in keywords if k and k.strip()]
    if not cleaned:
        return ""

    return ", ".join(cleaned[:limit])


def fetch_pair_sentiment(pair: str) -> Tuple[str, str]:
    """
    Fetch sentiment for a single currency pair from ForexNewsAPI /stat endpoint.

    Returns:
        (top_keywords_str, emoji)
    """
    if not FOREX_NEWS_API_KEY:
        logging.warning("FOREX_NEWS_API_KEY not set; returning empty sentiment")
        return "", "➖"

    api_symbol = pair_to_api_symbol(pair)
    params = {
        "currencypair": api_symbol,
        # NOTE: your plan currently complains about 'Stat section' access and 'date'.
        # We *do not* send 'date' or 'page' to keep requests as simple as possible.
        "token": FOREX_NEWS_API_KEY,
    }

    try:
        resp = requests.get(FOREX_NEWS_STAT_URL, params=params, timeout=10)
    except Exception as e:
        logging.warning(
            "Request exception (sentiment) for pair %s: %s",
            api_symbol,
            repr(e),
        )
        return "error", "➖"

    if not resp.ok:
        text = resp.text.strip()
        logging.warning(
            "HTTP error (sentiment) for pair %s: %s | body: %s",
            api_symbol,
            resp.status_code,
            text[:300],
        )
        # Mark as error but keep sheet consistent
        return "error", "➖"

    try:
        data = resp.json()
    except Exception as e:
        logging.warning(
            "JSON parse error for pair %s: %s | body: %s",
            api_symbol,
            repr(e),
            resp.text[:300],
        )
        return "error", "➖"

    # Try to locate the 'stat' block in different possible shapes
    stat = None
    if isinstance(data, dict):
        if "stat" in data:
            stat = data.get("stat")
        elif "result" in data and isinstance(data["result"], dict):
            stat = data["result"].get("stat")

    if stat is None:
        logging.warning(
        "No 'stat' section found in response for pair %s. Data keys: %s",
            api_symbol,
            list(data.keys()),
        )
        return "error", "➖"

    top_keywords = extract_top_keywords(stat, limit=5)
    emoji = classify_emoji_from_stat(stat)

    # If we truly have no keywords and sentiment is basically "no data"
    if not top_keywords and emoji == "➖":
        return "", "➖"

    return top_keywords, emoji


# -----------------------------------
# Sheet update logic (ONLY T & U)
# -----------------------------------
def update_sheet_news_sentiment(ws: gspread.Worksheet) -> None:
    """
    For each pair in FOREX_PAIRS, fetch sentiment and write it to columns:

        T = news_top_keywords_7d
        U = news_emoji_7d

    Only these two columns are touched. All other columns remain untouched.
    """
    num_pairs = len(FOREX_PAIRS)
    logging.info("Updating news sentiment (Sentiment endpoint) for %d pairs", num_pairs)

    # Ensure headers (T1:U1) are correct. This is harmless if they already exist.
    ws.update(
        NEWS_HEADER_RANGE,
        [[NEWS_TOP_KEYWORDS_COL_HEADER, NEWS_EMOJI_COL_HEADER]],
    )

    rows: List[List[Optional[str]]] = []

    for idx, pair in enumerate(FOREX_PAIRS, start=1):
        logging.info("(%d/%d) Fetching sentiment for %s", idx, num_pairs, pair)
        top_keywords, emoji = fetch_pair_sentiment(pair)
        rows.append([top_keywords, emoji])

    # Now write all values at once into T:U, rows aligned with FOREX_PAIRS.
    start_row = NEWS_DATA_START_ROW
    end_row = start_row + num_pairs - 1
    range_a1 = NEWS_DATA_COLUMNS_RANGE_TEMPLATE.format(start_row=start_row, end_row=end_row)

    logging.info("Updating sentiment columns %s", range_a1)
    ws.update(range_a1, rows)


# -----------------------------------
# Main loop
# -----------------------------------
def main() -> None:
    logging.info(
        "Starting Forex sentiment bot (Sentiment endpoint). "
        "Sheet='%s' Tab='%s' Interval=%ds",
        GOOGLE_SHEET_NAME,
        GOOGLE_SHEET_TAB,
        UPDATE_INTERVAL_SECONDS,
    )

    while True:
        try:
            ws = get_worksheet()
            update_sheet_news_sentiment(ws)
        except Exception as e:
            logging.exception("Unhandled error during sentiment update: %s", repr(e))

        logging.info("Sleeping for %d seconds...", UPDATE_INTERVAL_SECONDS)
        time.sleep(UPDATE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
