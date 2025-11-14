import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

import requests
import gspread
from google.oauth2.service_account import Credentials

# ---------------------------------------------------
# Config / ENV
# ---------------------------------------------------

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ForexNewsAPI base endpoint for currency pair news
# Docs example:
#   https://forexnewsapi.com/api/v1?currencypair=EUR-USD&items=10&token=YOUR_KEY
FOREX_NEWS_BASE_URL = os.getenv("FOREX_NEWS_BASE_URL", "https://forexnewsapi.com/api/v1")
FOREX_NEWS_API_KEY = os.getenv("FOREX_NEWS_API_KEY", "")

# How far back to look, using ForexNewsAPI &date= syntax (e.g. "last7days", "today")
FOREX_NEWS_DATE_PARAM = os.getenv("FOREX_NEWS_DATE", "last7days")

# Number of news items to pull per pair (1–50 per docs)
FOREX_NEWS_ITEMS = int(os.getenv("FOREX_NEWS_ITEMS", "30"))

SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Active-Investing")
SCREENER_TAB = os.getenv("OANDA_SCREENER_TAB", "Oanda-Screener")

# Interval between sentiment updates (seconds)
INTERVAL_SECONDS = int(os.getenv("SENTIMENT_INTERVAL_SECONDS", "3600"))  # default 1 hour

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Google Sheets helpers
# ---------------------------------------------------

def get_gspread_client() -> gspread.Client:
    """
    Authenticate to Google Sheets using service account JSON in GOOGLE_CREDS_JSON.
    """
    creds_json = os.environ["GOOGLE_CREDS_JSON"]
    info = json.loads(creds_json)
    credentials = Credentials.from_service_account_info(info, scopes=SCOPES)
    client = gspread.authorize(credentials)
    return client


def get_pairs_from_sheet(ws: gspread.Worksheet) -> List[str]:
    """
    Read the Oanda pairs from column A ('pair') of the screener tab.
    Stops at the first blank row.
    """
    col_vals = ws.col_values(1)  # column A
    if not col_vals:
        return []

    pairs: List[str] = []
    # Skip header (row 1)
    for val in col_vals[1:]:
        if not val:
            break
        pair = val.strip()
        if pair:
            pairs.append(pair)
    return pairs


def write_sentiment_to_sheet(
    ws: gspread.Worksheet,
    pairs: List[str],
    sentiment_rows: List[Dict[str, Any]],
) -> None:
    """
    Write sentiment columns starting at column T (20th column) on the screener tab.

    Columns written:
      T: news_sentiment_score_7d
      U: news_sentiment_label_7d
      V: news_articles_7d
      W: news_pos_7d
      X: news_neg_7d
      Y: news_neutral_7d
      Z: news_sample_headline_7d
    """

    header = [
        "news_sentiment_score_7d",
        "news_sentiment_label_7d",
        "news_articles_7d",
        "news_pos_7d",
        "news_neg_7d",
        "news_neutral_7d",
        "news_sample_headline_7d",
    ]

    values: List[List[Any]] = [header]

    for row in sentiment_rows:
        score = row.get("score")
        label = row.get("label") or ""
        total = row.get("total") or 0
        pos = row.get("pos") or 0
        neg = row.get("neg") or 0
        neu = row.get("neu") or 0
        sample_headline = row.get("sample_headline") or ""

        # Sheet wants primitive values; None -> ""
        score_cell = "" if score is None else round(float(score), 3)

        values.append([
            score_cell,
            label,
            total,
            pos,
            neg,
            neu,
            sample_headline,
        ])

    # Start at T1; this will fill T..Z (7 columns) for all rows given.
    ws.update("T1", values)
    logger.info(
        "Updated sentiment columns T:Z for %d instruments",
        len(sentiment_rows),
    )


# ---------------------------------------------------
# ForexNewsAPI sentiment helpers
# ---------------------------------------------------

def map_article_sentiment(s: str) -> Optional[int]:
    """
    Map ForexNewsAPI article sentiment string -> numeric.
    positive -> +1, negative -> -1, neutral -> 0.
    Unknown returns None.
    """
    if not s:
        return None
    s = s.strip().lower()
    if "positive" in s:
        return 1
    if "negative" in s:
        return -1
    if "neutral" in s:
        return 0
    return None


def label_from_score(score: Optional[float]) -> str:
    """
    Convert numeric score (~ -1..+1) to a label with emoji.
    """
    if score is None:
        return ""

    if score >= 0.5:
        return "😄 Strong Bullish"
    if score >= 0.2:
        return "🙂 Bullish"
    if score <= -0.5:
        return "😡 Strong Bearish"
    if score <= -0.2:
        return "🙁 Bearish"
    return "😐 Neutral"


def fetch_pair_sentiment(
    session: requests.Session,
    pair: str,
) -> Dict[str, Any]:
    """
    Fetch recent Forex news for a given Oanda pair and aggregate a simple sentiment score.

    pair example: 'EUR_USD' (Oanda style)
    ForexNewsAPI format: 'EUR-USD'
    """
    if not FOREX_NEWS_API_KEY:
        logger.warning("FOREX_NEWS_API_KEY not set; skipping sentiment for %s", pair)
        return {
            "pair": pair,
            "score": None,
            "label": "",
            "total": 0,
            "pos": 0,
            "neg": 0,
            "neu": 0,
            "sample_headline": "",
        }

    # Convert Oanda style to ForexNewsAPI style
    currencypair = pair.replace("_", "-")

    params = {
        "currencypair": currencypair,
        "items": max(1, min(FOREX_NEWS_ITEMS, 50)),
        "token": FOREX_NEWS_API_KEY,
        "date": FOREX_NEWS_DATE_PARAM,
    }

    try:
        resp = session.get(FOREX_NEWS_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Error fetching news for %s: %s", pair, exc)
        return {
            "pair": pair,
            "score": None,
            "label": "",
            "total": 0,
            "pos": 0,
            "neg": 0,
            "neu": 0,
            "sample_headline": "",
        }

    try:
        data = resp.json()
    except Exception as exc:
        logger.warning("Error parsing JSON for %s: %s", pair, exc)
        return {
            "pair": pair,
            "score": None,
            "label": "",
            "total": 0,
            "pos": 0,
            "neg": 0,
            "neu": 0,
            "sample_headline": "",
        }

    # ForexNewsAPI response is documented as JSON; structure can vary
    # We try a few common patterns:
    articles: List[Dict[str, Any]] = []
    if isinstance(data, list):
        articles = data
    elif isinstance(data, dict):
        if isinstance(data.get("data"), list):
            articles = data["data"]
        elif isinstance(data.get("news"), list):
            articles = data["news"]

    score_sum = 0
    count = 0
    pos = 0
    neg = 0
    neu = 0
    sample_headline = ""

    for idx, art in enumerate(articles):
        if not isinstance(art, dict):
            continue
        # ForexNewsAPI docs mention a sentiment field, e.g. "positive"/"negative"/"neutral"
        raw_sentiment = art.get("sentiment") or art.get("sentiment_label") or ""
        mapped = map_article_sentiment(raw_sentiment)
        if mapped is None:
            continue
        count += 1
        score_sum += mapped
        if mapped > 0:
            pos += 1
        elif mapped < 0:
            neg += 1
        else:
            neu += 1

        if not sample_headline:
            sample_headline = (art.get("title") or art.get("headline") or "").strip()

    if count == 0:
        score: Optional[float] = None
    else:
        score = score_sum / float(count)

    label = label_from_score(score)

    return {
        "pair": pair,
        "score": score,
        "label": label,
        "total": count,
        "pos": pos,
        "neg": neg,
        "neu": neu,
        "sample_headline": sample_headline,
    }


# ---------------------------------------------------
# Main sentiment update logic
# ---------------------------------------------------

def run_sentiment_update():
    gc = get_gspread_client()
    sh = gc.open(SHEET_NAME)
    ws = sh.worksheet(SCREENER_TAB)

    pairs = get_pairs_from_sheet(ws)
    if not pairs:
        logger.warning("No pairs found in '%s'!A", SCREENER_TAB)
        return

    logger.info("Updating news sentiment for %d pairs", len(pairs))

    session = requests.Session()
    sentiment_rows: List[Dict[str, Any]] = []

    for idx, pair in enumerate(pairs, start=1):
        logger.info("(%d/%d) Fetching sentiment for %s", idx, len(pairs), pair)
        row = fetch_pair_sentiment(session, pair)
        sentiment_rows.append(row)
        # Gentle throttle in case of rate limits
        time.sleep(0.3)

    write_sentiment_to_sheet(ws, pairs, sentiment_rows)


# ---------------------------------------------------
# Main loop
# ---------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info(
        "Starting Forex sentiment bot. Sheet='%s' Tab='%s' Interval=%ss",
        SHEET_NAME,
        SCREENER_TAB,
        INTERVAL_SECONDS,
    )

    while True:
        try:
            run_sentiment_update()
        except Exception as exc:
            logger.exception("Error in sentiment loop: %s", exc)
        logger.info("Sleeping for %s seconds before next sentiment update...", INTERVAL_SECONDS)
        time.sleep(INTERVAL_SECONDS)
