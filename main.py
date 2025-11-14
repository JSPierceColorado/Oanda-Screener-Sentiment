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

# Flag so we only dump one sample payload per run
PRINTED_SAMPLE = False

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
        "User-Agent": "Aletheia-ForexSentimentBot/1.2"
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

    Expected pattern (per docs):
        GET {sentiment_url}?currencypair=EUR-USD&date=last7days&token=YOUR_KEY

    We convert OANDA style 'EUR_USD' -> 'EUR-USD'.

    We defensively handle shapes like:
      - {"sentimentscore": ...}
      - {"data": [{...}]}
      - {"EUR-USD": {"sentimentscore": ...}, ...}
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

    item: Optional[Dict[str, Any]] = None

    # ----- Case 1: dict keyed by pair, e.g. {"EUR-USD": {...}, "GBP-USD": {...}} -----
    if isinstance(data, dict):
        lower_keys = {k.lower(): k for k in data.keys()}

        # Direct match on pair (case-insensitive)
        key_exact = lower_keys.get(pair_fx.lower())
        if key_exact and isinstance(data.get(key_exact), dict):
            item = data[key_exact]
        else:
            # Common container keys
            for key in ("data", "results", "sentiment"):
                v = data.get(key)
                if isinstance(v, dict):
                    inner_lk = {kk.lower(): kk for kk in v.keys()}
                    k2 = inner_lk.get(pair_fx.lower())
                    if k2 and isinstance(v.get(k2), dict):
                        item = v[k2]
                        break
                if isinstance(v, list) and v:
                    if isinstance(v[0], dict):
                        item = v[0]
                        break

            # Fallback: maybe it's already a single-pair dict like {"sentimentscore": ...}
            if item is None:
                if any("sentiment" in k.lower() or "score" in k.lower() for k in data.keys()):
                    item = data

    # ----- Case 2: top-level list of dicts -----
    if item is None and isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            item = first

    if not isinstance(item, dict):
        logger.warning("Unexpected sentiment payload for pair %s: %r", pair_code_oanda, data)
        return None

    global PRINTED_SAMPLE
    if not PRINTED_SAMPLE:
        # Log a single sample item so you can see raw structure in Railway logs
        trimmed = {k: item[k] for k in list(item.keys())[:10]}  # first 10 keys
        logger.info("Sample sentiment payload for %s: %r", pair_code_oanda, trimmed)
        PRINTED_SAMPLE = True

    return item


def normalize_sentiment_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize ForexNewsAPI sentiment response into a uniform dict we can write to the sheet.

    Sentiment Score ranges from -1.5 (Negative) to +1.5 (Positive) per docs.
    We try to detect:
      - sentiment score (numeric OR text)
      - positive / negative / neutral counts if available
      - date of the sentiment value
    """

    # ---------- Sentiment score (numeric) ----------
    score_raw = (
        item.get("sentimentscore")
        or item.get("sentiment_score")
        or item.get("score")
        or item.get("sentimentScore")
    )

    # Generic numeric key scan if not found
    if score_raw is None:
        for k, v in item.items():
            kl = k.lower()
            if ("sentiment" in kl or "score" in kl) and v not in (None, ""):
                try:
                    score_raw = float(v)
                    break
                except (TypeError, ValueError):
                    continue

    try:
        score = float(score_raw) if score_raw is not None else None
    except (TypeError, ValueError):
        score = None

    # ---------- Sentiment text (positive/negative/neutral) ----------
    sentiment_text = (
        item.get("sentiment")
        or item.get("newsentiment")
        or item.get("sentiment_label")
        or item.get("label")
    )

    if isinstance(sentiment_text, str):
        st = sentiment_text.strip().lower()
        if "positive" in st or st == "pos":
            if score is None:
                score = 1.0
        elif "negative" in st or st == "neg":
            if score is None:
                score = -1.0
        elif "neutral" in st or st == "neu":
            if score is None:
                score = 0.0

    # ---------- Counts (positive/negative/neutral/total) ----------
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
        or item.get("total_articles")
        or item.get("num_articles")
        or item.get("articles")
    )

    def pick_int_like(key_substrings: List[str]) -> Optional[int]:
        for k, v in item.items():
            kl = k.lower()
            if any(s in kl for s in key_substrings):
                try:
                    return int(v)
                except (TypeError, ValueError):
                    continue
        return None

    def to_int_safe(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    pos_i = to_int_safe(pos) or pick_int_like(["positive"])
    neg_i = to_int_safe(neg) or pick_int_like(["negative"])
    neu_i = to_int_safe(neu) or pick_int_like(["neutral"])
    total_i = to_int_safe(total) or pick_int_like(["total", "articles", "news"])

    # If total isn't provided, approximate from pos/neg/neu
    if total_i is None:
        parts = [v for v in (pos_i, neg_i, neu_i) if v is not None]
        total_i = sum(parts) if parts else None

    # If still no score but we have counts, derive a score in [-1.5, 1.5]
    if score is None and total_i and total_i > 0:
        pos_val = pos_i or 0
        neg_val = neg_i or 0
        score = 1.5 * (pos_val - neg_val) / float(total_i)

    # Percentages if we have total
    if total_i and total_i > 0:
        pos_pct = round((pos_i or 0) / total_i * 100.0, 1)
        neg_pct = round((neg_i or 0) / total_i * 100.0, 1)
    else:
        pos_pct = ""
        neg_pct = ""

    # ---------- Date / as-of field ----------
    sent_date = (
        item.get("date")
        or item.get("sentiment_date")
        or item.get("asof")
        or item.get("as_of")
    )

    if not sent_date:
        for k, v in item.items():
            kl = k.lower()
            if "date" in kl or "time" in kl:
                sent_date = v
                break

    # ---------- Label + emoji ----------
    if score is None:
        label = sentiment_text or ""
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
        "label": label or "",
        "emoji": emoji,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "total": total_i if total_i is not None else "",
        "date": sent_date or "",
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
    ws.update("T1", [headers])

    max_row = max(row_results.keys())
    values: List[List[Any]] = []

    for row_idx in range(2, max_row + 1):
        sentiment = row_results.get(row_idx)
        if not sentiment:
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
    ws.update(range_name, values)


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------


def run_sentiment_once():
    sheet_name = os.environ.get("GOOGLE_SHEET_NAME", "Active-Investing")
    tab_name = os.environ.get("GOOGLE_TAB_NAME", "Oanda-Screener")

    api_key = os.environ.get("FOREX_NEWS_API_KEY")
    if not api_key:
        raise RuntimeError("FOREX_NEWS_API_KEY env var is required")

    # Individual pair sentiment endpoint URL,
    # e.g. https://forexnewsapi.com/api/v1/sentiment-individual  (whatever your docs show)
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
