import os
import time
import json
import logging
from typing import Optional, Tuple, List

import requests
import gspread
from google.oauth2.service_account import Credentials

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------------------------------------------------------
# Environment / config
# -----------------------------------------------------------------------------
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")
GOOGLE_SHEET_NAME = os.getenv("FOREX_SHEET_NAME", "Active-Investing")
FOREX_SENTIMENT_TAB = os.getenv("FOREX_SENTIMENT_TAB", "Oanda-Screener")
FOREX_NEWS_API_KEY = os.getenv("FOREX_NEWS_API_KEY")

# Optional: if you later upgrade your ForexNewsAPI plan to support &date=
# you can set this env var (e.g. "last7days", "last30days"). For the current
# plan we must NOT send any date parameter or we get 403.
FOREX_NEWS_DATE = (os.getenv("FOREX_NEWS_DATE") or "").strip() or None

# How often to refresh sentiment (seconds)
FOREX_SENTIMENT_INTERVAL = int(os.getenv("FOREX_SENTIMENT_INTERVAL", "21600"))  # 6h default

# gspread scopes (include Drive so we can open by name)
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Sheet layout:
# Col A: Pair (e.g. "EUR_USD")
# We will write sentiment into columns T:Z (20–26)
SENTIMENT_FIRST_COL = 20  # Column T
SENTIMENT_LAST_COL = 26   # Column Z
FIRST_DATA_ROW = 2        # Header is row 1


# -----------------------------------------------------------------------------
# Google Sheets helpers
# -----------------------------------------------------------------------------
def get_gspread_client() -> gspread.Client:
    if not GOOGLE_CREDS_JSON:
        raise RuntimeError("GOOGLE_CREDS_JSON env var is not set")

    info = json.loads(GOOGLE_CREDS_JSON)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client


def get_worksheet() -> gspread.Worksheet:
    client = get_gspread_client()
    sh = client.open(GOOGLE_SHEET_NAME)
    ws = sh.worksheet(FOREX_SENTIMENT_TAB)
    return ws


# -----------------------------------------------------------------------------
# ForexNewsAPI sentiment fetching
# -----------------------------------------------------------------------------
def fetch_pair_sentiment(
    session: requests.Session,
    api_key: str,
    pair_code: str,
) -> Optional[Tuple[float, float, float, float, int, str]]:
    """
    Fetch sentiment stats for a single FX pair from ForexNewsAPI.

    Returns:
        (avg_score, pos_pct, neg_pct, neu_pct, article_count, status_text)
        or None if there was an HTTP / parsing error.
    """
    base_url = "https://forexnewsapi.com/api/v1/stat"

    # IMPORTANT: the current subscription plan returns 403 if we send &date=
    # ("date is not available with current Subscription Plan.")
    # So by default we do NOT send date at all.
    params = {
        "currencypair": pair_code,
        "token": api_key,
    }
    if FOREX_NEWS_DATE:
        # Only include if explicitly configured and the plan supports it.
        params["date"] = FOREX_NEWS_DATE

    try:
        resp = session.get(base_url, params=params, timeout=10)
    except requests.RequestException as e:
        logging.warning("Network error fetching sentiment for %s: %s", pair_code, e)
        return None

    if resp.status_code != 200:
        body_snip = resp.text.strip()
        if len(body_snip) > 200:
            body_snip = body_snip[:200] + "..."
        logging.warning(
            "HTTP error (sentiment) for pair %s: %s | body: %s",
            pair_code,
            resp.status_code,
            body_snip,
        )
        return None

    try:
        data = resp.json()
    except Exception as e:
        logging.warning("JSON parse error for %s: %s | body: %s", pair_code, e, resp.text[:200])
        return None

    # The exact schema can vary; we try to be defensive.
    # Typical shape (from docs) is something like:
    # {
    #   "data": [
    #      {
    #         "date": "2024-01-01",
    #         "sentiment": {
    #             "score": 0.12,
    #             "positive": 60,
    #             "negative": 20,
    #             "neutral": 20
    #         },
    #         "news": 15
    #      },
    #      ...
    #   ]
    # }
    stats = data.get("data") or data.get("stats") or []

    if not stats:
        return 0.0, 0.0, 0.0, 0.0, 0, "no-data"

    total_score = 0.0
    total_pos = 0.0
    total_neg = 0.0
    total_neu = 0.0
    total_news = 0
    days_count = 0

    for day in stats:
        sent = day.get("sentiment") or {}
        score = float(sent.get("score", 0.0) or 0.0)
        pos = float(sent.get("positive", 0.0) or 0.0)
        neg = float(sent.get("negative", 0.0) or 0.0)
        neu = float(sent.get("neutral", 0.0) or 0.0)
        news_count = int(day.get("news", 0) or 0)

        total_score += score
        total_pos += pos
        total_neg += neg
        total_neu += neu
        total_news += news_count
        days_count += 1

    if days_count == 0:
        return 0.0, 0.0, 0.0, 0.0, total_news, "no-days"

    avg_score = total_score / days_count
    avg_pos = total_pos / days_count
    avg_neg = total_neg / days_count
    avg_neu = total_neu / days_count

    status_text = f"ok:{days_count}d"
    return avg_score, avg_pos, avg_neg, avg_neu, total_news, status_text


# -----------------------------------------------------------------------------
# Main sheet update logic
# -----------------------------------------------------------------------------
def update_sheet_sentiment() -> None:
    if not FOREX_NEWS_API_KEY:
        logging.error("FOREX_NEWS_API_KEY is not set. Exiting sentiment update.")
        return

    ws = get_worksheet()

    # Get all pair symbols from column A (starting row 2)
    pairs_col = ws.col_values(1)  # 1 = column A
    # First element is header; from second onward are pairs
    pair_rows: List[Tuple[int, str]] = []
    for idx, val in enumerate(pairs_col, start=1):
        if idx < FIRST_DATA_ROW:
            continue
        val = (val or "").strip()
        if not val:
            continue
        pair_rows.append((idx, val))

    if not pair_rows:
        logging.info("No pairs found in column A; nothing to update.")
        return

    logging.info("Updating news sentiment (Sentiment endpoint) for %d pairs", len(pair_rows))

    session = requests.Session()

    # Prepare rows for T..Z, aligned to pair_rows
    # Columns:
    # T: AvgScore
    # U: AvgPos%
    # V: AvgNeg%
    # W: AvgNeu%
    # X: TotalNewsCount
    # Y: LastUpdated (ISO string)
    # Z: Status / debug
    values: List[List] = []

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for i, (row_idx, pair_name) in enumerate(pair_rows, start=1):
        pair_code = pair_name.replace("_", "-")  # e.g. EUR_USD -> EUR-USD
        logging.info("(%d/%d) Fetching sentiment for %s", i, len(pair_rows), pair_name)

        result = fetch_pair_sentiment(session, FOREX_NEWS_API_KEY, pair_code)

        if result is None:
            # HTTP / parse error: leave numeric fields blank, status with error flag
            values.append(["", "", "", "", "", ts, "error"])
        else:
            avg_score, avg_pos, avg_neg, avg_neu, total_news, status_text = result
            values.append([
                round(avg_score, 4),
                round(avg_pos, 2),
                round(avg_neg, 2),
                round(avg_neu, 2),
                total_news,
                ts,
                status_text,
            ])

    # Write to T2:Z{last_row}
    first_row = FIRST_DATA_ROW
    last_row = pair_rows[-1][0]
    range_str = f"T{first_row}:Z{last_row}"
    logging.info("Updating sentiment columns %s", range_str)

    # NOTE: gspread deprecated old arg order; we now pass values first, then range_name.
    ws.update(values, range_name=range_str)

    logging.info("Sentiment update complete for %d rows", len(pair_rows))


# -----------------------------------------------------------------------------
# Entrypoint loop
# -----------------------------------------------------------------------------
def main():
    logging.info(
        "Starting Forex sentiment bot (Sentiment endpoint). "
        "Sheet='%s' Tab='%s' Interval=%ss",
        GOOGLE_SHEET_NAME,
        FOREX_SENTIMENT_TAB,
        FOREX_SENTIMENT_INTERVAL,
    )

    while True:
        try:
            update_sheet_sentiment()
        except Exception as e:
            logging.exception("Unexpected error during sentiment update: %s", e)

        logging.info("Sleeping for %s seconds...", FOREX_SENTIMENT_INTERVAL)
        time.sleep(FOREX_SENTIMENT_INTERVAL)


if __name__ == "__main__":
    main()
