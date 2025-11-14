import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("forex-sentiment-bot")

# --------------------------------------------------------------------------------------
# Google Sheets helpers
# --------------------------------------------------------------------------------------


def get_gspread_client() -> gspread.Client:
    google_creds_json = os.environ.get("GOOGLE_CREDS_JSON")
    if not google_creds_json:
        raise RuntimeError("GOOGLE_CREDS_JSON env var is not set")

    creds_dict = json.loads(google_creds_json)

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
    ]

    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
        creds_dict, scopes=scope
    )
    gc = gspread.authorize(credentials)
    return gc


def open_screener_sheet(
    gc: gspread.Client,
    sheet_name: str,
    tab_name: str
) -> gspread.Worksheet:
    sh = gc.open(sheet_name)
    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        raise RuntimeError(f"Worksheet '{tab_name}' not found in '{sheet_name}'")
    return ws


def read_pairs_from_sheet(ws: gspread.Worksheet) -> List[Tuple[int, str]]:
    """
    Read instrument codes from column A (starting row 2) until
    we hit an empty cell. Returns list of (row_index, instrument).
    """
    col_values = ws.col_values(1)  # 1-based, column A
    pairs: List[Tuple[int, str]] = []
    # col_values[0] is header; start from index 1
    for idx, val in enumerate(col_values[1:], start=2):
        if not val:
            break
        pairs.append((idx, val.strip()))
    return pairs


# --------------------------------------------------------------------------------------
# ForexNewsAPI Sentiment helpers
# --------------------------------------------------------------------------------------


def create_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Accept": "application/json",
        "User-Agent": "Aletheia-ForexSentimentBot/1.0"
    })
    return s


def fetch_sentiment_for_pair(
    session: requests.Session,
    sentiment_url: str,
    api_key: str,
    pair_code_oanda: str,
    date_range: str,
    timeout: int = 15,
) -> Optional[Dict[str, Any]]:
    """
    Fetch sentiment for a currency pair from ForexNewsAPI's Sentiment Analysis endpoint.

    Expected pattern (you'll confirm in your docs):
        GET {sentiment_url}?currencypair=EUR-USD&date=last7days&token=YOUR_KEY

    We convert OANDA style 'EUR_USD' -> 'EUR-USD'.
    """
    pair_fx = pair_code_oanda.replace("_", "-")

    params = {
        "currencypair": pair_fx,
        "date": date_range,
        "token": api_key,
    }

    try:
        resp = session.get(sentiment_url, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.HTTPError as e:
        logger.warning(
            "HTTP error (sentiment) for pair %s (%s): %s",
            pair_code_oanda,
            pair_fx,
            e
        )
        return None
    except requests.RequestException as e:
        logger.warning(
            "Request exception (sentiment) for pair %s (%s): %s",
            pair_code_oanda,
            pair_fx,
            e
        )
        return None

    try:
        data = resp.json()
    except ValueError:
        logger.warning("Invalid JSON (sentiment) for pair %s (%s)", pair_code_oanda, pair_fx)
        return None

    # ForexNewsAPI sentiment endpoints may return a dict or list.
    # We'll try a few reasonable shapes:
    item: Optional[Dict[str, Any]] = None

    if isinstance(data, dict):
        # e.g. {'currencypair': 'EUR-USD', 'sentimentscore': 0.85, ...}
        item = data
        # or sometimes under a key like 'data' / 'results'
        for key in ("data", "results", "sentiment"):
            if isinstance(data.get(key), dict):
                item = data[key]
                break
            if isinstance(data.get(key), list) and data[key]:
                item = data[key][0]
                break
    elif isinstance(data, list) and data:
        item = data[0]

    if not isinstance(item, dict):
        logger.warning("Unexpected sentiment payload for pair %s: %r", pair_code_oanda, data)
        return None

    return item


def normalize_sentiment_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize ForexNewsAPI sentiment response into a uniform dict we can write to the sheet.

    Their docs mention Sentiment Score ranges from -1.5 (Negative) to +1.5 (Positive). :contentReference[oaicite:1]{index=1}
    We'll try to detect:
      - sentiment score (numeric)
      - positive / negative / neutral counts if available
      - date of the sentiment value
    """

    # Try several likely keys for the numeric score
    score_raw = (
        item.get("sentimentscore")
        or item.get("sentiment_score")
        or item.get("score")
        or item.get("sentimentScore")
    )

    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        score = None

    # Optional counts / totals (these may or may not exist)
    pos = (
        item.get("positive")
        or item.get("pos")
        or item.get("num_positive")
        or item.get("count_positive")
    )
    neg = (
        item.get("negative")
        or item.get("neg")
        or item.get("num_negative")
        or item.get("count_negative")
    )
    neu = (
        item.get("neutral")
        or item.get("neu")
        or item.get("num_neutral")
        or item.get("count_neutral")
    )
    total = (
        item.get("total")
        or item.get("total_news")
        or item.get("num_articles")
        or item.get("articles")
    )

    def _to_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    pos_i = _to_int(pos)
    neg_i = _to_int(neg)
    neu_i = _to_int(neu)
    total_i = _to_int(total)

    # If total isn't provided, approximate from pos/neg/neu
    if total_i is None:
        parts = [v for v in (pos_i, neg_i, neu_i) if v is not None]
        total_i = sum(parts) if parts else None

    # Percentages if we have total
    if total_i and total_i > 0:
        pos_pct = round((pos_i or 0) / total_i * 100.0, 1)
        neg_pct = round((neg_i or 0) / total_i * 100.0, 1)
    else:
        pos_pct = ""
        neg_pct = ""

    # Date/time field for when this sentiment is measured
    sent_date = (
        item.get("date")
        or item.get("sentiment_date")
        or item.get("asof")
        or ""
    )

    # Label + emoji from score
    if score is None:
        label = ""
        emoji = ""
    else:
        if score >= 1.0:
            label, emoji = "Strongly Bullish", "🟢"
        elif score >= 0.3:
            label, emoji = "Bullish", "✅"
        elif score <= -1.0:
            label, emoji = "Strongly Bearish", "🔴"
        elif score <= -0.3:
            label, emoji = "Bearish", "⚠️"
        else:
            label, emoji = "Neutral", "⚪"

    if score is not None:
        score = round(score, 3)

    return {
        "score": score,
        "label": label,
        "emoji": emoji,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "total": total_i if total_i is not None else "",
        "date": sent_date,
    }


# --------------------------------------------------------------------------------------
# Writing results to sheet (columns T–Z)
# --------------------------------------------------------------------------------------


def write_sentiment_to_sheet(
    ws: gspread.Worksheet,
    row_results: Dict[int, Dict[str, Any]],
):
    """
    row_results: {row_index: sentiment_dict}

    Columns we use (T–Z):

    T: News_Sent_Score_7d      (numeric, -1.5..+1.5-ish)
    U: News_Sent_Label_7d      (text like "Bullish")
    V: News_Sent_Emoji_7d      (emoji)
    W: News_Pos_Pct_7d         (% positive, if available)
    X: News_Neg_Pct_7d         (% negative, if available)
    Y: News_Total_Articles_7d  (if available)
    Z: News_Sentiment_Date_7d  (date/as-of from API)
    """
    if not row_results:
        logger.info("No sentiment results to write.")
        return

    headers = [
        "News_Sent_Score_7d",
        "News_Sent_Label_7d",
        "News_Sent_Emoji_7d",
        "News_Pos_Pct_7d",
        "News_Neg_Pct_7d",
        "News_Total_Articles_7d",
        "News_Sentiment_Date_7d",
    ]

    # Write headers into T1:Z1 (7 columns)
    ws.update(
        range_name="T1:Z1",
        values=[headers],
    )

    max_row = max(row_results.keys())
    values: List[List[Any]] = []

    for row_idx in range(2, max_row + 1):
        sentiment = row_results.get(row_idx)
        if not sentiment:
            # Keep existing cells untouched by sending blanks
            values.append([""] * len(headers))
            continue

        score = sentiment["score"]
        label = sentiment["label"]
        emoji = sentiment["emoji"]
        pos_pct = sentiment["pos_pct"]
        neg_pct = sentiment["neg_pct"]
        total = sentiment["total"]
        date = sentiment["date"]

        row_vals = [
            score if score is not None else "",
            label,
            emoji,
            pos_pct,
            neg_pct,
            total,
            date,
        ]

        values.append(row_vals)

    end_row = max_row
    range_name = f"T2:Z{end_row}"
    logger.info("Updating sentiment columns %s", range_name)
    ws.update(
        range_name=range_name,
        values=values,
    )


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------


def run_sentiment_once():
    sheet_name = os.environ.get("GOOGLE_SHEET_NAME", "Active-Investing")
    tab_name = os.environ.get("GOOGLE_TAB_NAME", "Oanda-Screener")

    api_key = os.environ.get("FOREX_NEWS_API_KEY")
    if not api_key:
        raise RuntimeError("FOREX_NEWS_API_KEY env var is required")

    # IMPORTANT:
    # This should be the Sentiment Analysis endpoint URL for INDIVIDUAL pairs,
    # as shown under "GET Sentiment Analysis" -> "For Individual currency pairs" in the docs.
    # Example guess (override if docs show different):
    #   https://forexnewsapi.com/api/v1/sentiment
    sentiment_url = os.environ.get(
        "FOREX_NEWS_SENTIMENT_URL",
        "https://forexnewsapi.com/api/v1/sentiment",
    )

    # date window for sentiment, e.g. 'last7days', 'last30days', 'today', etc.
    date_range = os.environ.get("FOREX_NEWS_DATE", "last7days")

    gc = get_gspread_client()
    ws = open_screener_sheet(gc, sheet_name, tab_name)

    pairs = read_pairs_from_sheet(ws)
    logger.info("Updating news sentiment (Sentiment endpoint) for %d pairs", len(pairs))

    session = create_session()
    row_results: Dict[int, Dict[str, Any]] = {}

    for idx, (row_idx, pair_code) in enumerate(pairs, start=1):
        logger.info("(%d/%d) Fetching sentiment for %s", idx, len(pairs), pair_code)

        raw_item = fetch_sentiment_for_pair(
            session=session,
            sentiment_url=sentiment_url,
            api_key=api_key,
            pair_code_oanda=pair_code,
            date_range=date_range,
        )

        if not raw_item:
            continue

        sentiment = normalize_sentiment_item(raw_item)
        row_results[row_idx] = sentiment

        time.sleep(0.2)

    write_sentiment_to_sheet(ws, row_results)


def main_loop():
    sheet_name = os.environ.get("GOOGLE_SHEET_NAME", "Active-Investing")
    tab_name = os.environ.get("GOOGLE_TAB_NAME", "Oanda-Screener")

    interval_seconds = int(
        os.environ.get("FOREX_NEWS_INTERVAL_SECONDS", str(24 * 3600))
    )  # default: once per day

    logger.info(
        "Starting Forex sentiment bot (Sentiment endpoint). Sheet='%s' Tab='%s' Interval=%ss",
        sheet_name,
        tab_name,
        interval_seconds,
    )

    while True:
        try:
            run_sentiment_once()
        except Exception as e:
            logger.exception("Error in sentiment loop: %s", e)

        logger.info("Sleeping for %d seconds...", interval_seconds)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main_loop()
