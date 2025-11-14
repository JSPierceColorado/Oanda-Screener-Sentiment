import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

import requests
import gspread
import pandas as pd
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
# ForexNewsAPI helpers
# --------------------------------------------------------------------------------------


def create_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Accept": "application/json",
        "User-Agent": "Aletheia-ForexSentimentBot/1.0"
    })
    return s


def fetch_news_for_pair(
    session: requests.Session,
    base_url: str,
    api_key: str,
    pair_code_oanda: str,
    items: int,
    date_range: str,
    timeout: int = 15,
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch latest news for a currency pair from ForexNewsAPI.

    Docs example: GET https://forexnewsapi.com/api/v1?currencypair=EUR-USD&items=10&token=YOUR_KEY
    We convert OANDA style 'EUR_USD' -> 'EUR-USD'.
    """
    pair_fx = pair_code_oanda.replace("_", "-")

    params = {
        "currencypair": pair_fx,
        "items": items,
        "token": api_key,
        # date can be 'last7days', 'today', 'last30days', etc. :contentReference[oaicite:1]{index=1}
        "date": date_range,
    }

    try:
        resp = session.get(base_url, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.HTTPError as e:
        logger.warning(
            "HTTP error for pair %s (%s): %s",
            pair_code_oanda,
            pair_fx,
            e
        )
        return None
    except requests.RequestException as e:
        logger.warning(
            "Request exception for pair %s (%s): %s",
            pair_code_oanda,
            pair_fx,
            e
        )
        return None

    try:
        data = resp.json()
    except ValueError:
        logger.warning("Invalid JSON for pair %s (%s)", pair_code_oanda, pair_fx)
        return None

    # ForexNewsAPI usually returns a list of news items directly, or under a key.
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try common keys
        for key in ("data", "news", "items", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
    # Fallback
    return None


def aggregate_sentiment_from_news(
    news_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Take article-level news with sentiment labels and aggregate metrics.

    We assume each item has a 'sentiment' field that is 'positive', 'negative' or 'neutral'. :contentReference[oaicite:2]{index=2}
    If ForexNewsAPI returns a numeric sentiment field too, we try to use it if present.
    """

    pos = neg = neu = 0
    score_sum = 0.0
    score_count = 0

    latest_time = None
    latest_title = ""
    latest_source = ""

    for item in news_items:
        sent = str(item.get("sentiment", "")).lower()
        # Optional numeric score field (if present)
        numeric_sent = item.get("sentimentscore") or item.get("sentiment_score")

        if numeric_sent is not None:
            try:
                numeric_sent = float(numeric_sent)
                score_sum += numeric_sent
                score_count += 1
            except (TypeError, ValueError):
                pass
        else:
            # Fall back to +1 / -1 / 0
            if sent == "positive":
                score_sum += 1.0
                pos += 1
                score_count += 1
            elif sent == "negative":
                score_sum -= 1.0
                neg += 1
                score_count += 1
            elif sent == "neutral":
                neu += 1
                score_count += 1
            else:
                # Unknown / missing sentiment -> ignore
                pass

        # Track "latest" by published date if present, else just first item
        item_time = item.get("date") or item.get("published_at") or item.get("time")
        if latest_time is None and item_time:
            latest_time = item_time
            latest_title = str(item.get("title", ""))[:200]
            latest_source = str(item.get("source", ""))[:100]

    if score_count == 0:
        avg_score = ""
    else:
        avg_score = round(score_sum / score_count, 3)

    total = pos + neg + neu
    if total == 0:
        pos_pct = neg_pct = neu_pct = ""
    else:
        pos_pct = round(pos / total * 100.0, 1)
        neg_pct = round(neg / total * 100.0, 1)
        neu_pct = round(neu / total * 100.0, 1)

    # Map average numeric score to a coarse label/emoji
    label = ""
    emoji = ""
    if isinstance(avg_score, float):
        if avg_score >= 0.6:
            label, emoji = "Strongly Bullish", "🟢"
        elif avg_score >= 0.2:
            label, emoji = "Bullish", "✅"
        elif avg_score > -0.2:
            label, emoji = "Neutral", "⚪"
        elif avg_score > -0.6:
            label, emoji = "Bearish", "⚠️"
        else:
            label, emoji = "Strongly Bearish", "🔴"

    return {
        "avg_score": avg_score,
        "label": label,
        "emoji": emoji,
        "pos": pos,
        "neg": neg,
        "neu": neu,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "neu_pct": neu_pct,
        "total": total,
        "latest_time": latest_time or "",
        "latest_title": latest_title,
        "latest_source": latest_source,
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

    Columns we use:

    T: News_Sent_Score_7d      (numeric, -1..+1-ish)
    U: News_Sent_Label_7d      (text like "Bullish")
    V: News_Sent_Emoji_7d      (emoji)
    W: News_Pos_Pct_7d         (% positive)
    X: News_Neg_Pct_7d         (% negative)
    Y: News_Total_Articles_7d
    Z: News_Last_Update        (source + time)
    """
    if not row_results:
        logger.info("No sentiment results to write.")
        return

    # Header row
    headers = [
        "News_Sent_Score_7d",
        "News_Sent_Label_7d",
        "News_Sent_Emoji_7d",
        "News_Pos_Pct_7d",
        "News_Neg_Pct_7d",
        "News_Total_Articles_7d",
        "News_Last_Update",
    ]

    # Write headers into T1:Z1 (7 columns)
    ws.update(
        range_name="T1:Z1",
        values=[headers],
    )

    # Build values list for rows 2..N
    max_row = max(row_results.keys())
    values: List[List[Any]] = []

    for row_idx in range(2, max_row + 1):
        sentiment = row_results.get(row_idx)
        if not sentiment:
            # Keep existing cells untouched by sending blanks
            values.append([""] * len(headers))
            continue

        avg_score = sentiment["avg_score"]
        label = sentiment["label"]
        emoji = sentiment["emoji"]
        pos_pct = sentiment["pos_pct"]
        neg_pct = sentiment["neg_pct"]
        total = sentiment["total"]
        latest_time = sentiment["latest_time"]
        latest_source = sentiment["latest_source"]

        last_update = ""
        if latest_time or latest_source:
            last_update = f"{latest_source} | {latest_time}".strip(" |")

        row_vals = [
            avg_score,
            label,
            emoji,
            pos_pct,
            neg_pct,
            total,
            last_update,
        ]

        values.append(row_vals)

    # Now write into T2:Z{max_row}
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

    base_url = os.environ.get(
        "FOREX_NEWS_BASE_URL",
        "https://forexnewsapi.com/api/v1",
    )

    # how many articles per pair (1–50 per docs) :contentReference[oaicite:3]{index=3}
    items = int(os.environ.get("FOREX_NEWS_ITEMS", "20"))

    # date window for news, e.g. 'last7days', 'last30days', 'today', etc. :contentReference[oaicite:4]{index=4}
    date_range = os.environ.get("FOREX_NEWS_DATE", "last7days")

    gc = get_gspread_client()
    ws = open_screener_sheet(gc, sheet_name, tab_name)

    pairs = read_pairs_from_sheet(ws)
    logger.info("Updating news sentiment for %d pairs", len(pairs))

    session = create_session()

    row_results: Dict[int, Dict[str, Any]] = {}

    for idx, (row_idx, pair_code) in enumerate(pairs, start=1):
        logger.info("(%d/%d) Fetching sentiment for %s", idx, len(pairs), pair_code)

        news_items = fetch_news_for_pair(
            session=session,
            base_url=base_url,
            api_key=api_key,
            pair_code_oanda=pair_code,
            items=items,
            date_range=date_range,
        )

        if not news_items:
            # No news / error -> leave row untouched
            continue

        sentiment = aggregate_sentiment_from_news(news_items)
        row_results[row_idx] = sentiment

        # Tiny sleep to be polite to API
        time.sleep(0.2)

    write_sentiment_to_sheet(ws, row_results)


def main_loop():
    sheet_name = os.environ.get("GOOGLE_SHEET_NAME", "Active-Investing")
    tab_name = os.environ.get("GOOGLE_TAB_NAME", "Oanda-Screener")

    interval_seconds = int(
        os.environ.get("FOREX_NEWS_INTERVAL_SECONDS", str(24 * 3600))
    )  # default: once per day

    logger.info(
        "Starting Forex sentiment bot. Sheet='%s' Tab='%s' Interval=%ss",
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
