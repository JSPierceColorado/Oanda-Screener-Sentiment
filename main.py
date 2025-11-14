import os
import json
import time
import math
import logging
from typing import Any, Dict, List

import requests
import gspread
from google.oauth2.service_account import Credentials

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Active-Investing")
GOOGLE_SHEET_TAB = os.getenv("GOOGLE_SHEET_TAB", "Oanda-Screener")

FOREX_NEWS_API_KEY = os.getenv("FOREX_NEWS_API_KEY", "")
# base endpoint – SHOULD BE: https://forexnewsapi.com/api/v1/stat
FOREX_NEWS_SENTIMENT_URL = os.getenv(
    "FOREX_NEWS_SENTIMENT_URL", "https://forexnewsapi.com/api/v1/stat"
)
FOREX_NEWS_DATE = os.getenv("FOREX_NEWS_DATE", "last7days")
FOREX_SENTIMENT_INTERVAL = int(os.getenv("FOREX_SENTIMENT_INTERVAL", "21600"))

if not GOOGLE_CREDS_JSON:
    logger.error("GOOGLE_CREDS_JSON env var is required")
if not FOREX_NEWS_API_KEY:
    logger.warning("FOREX_NEWS_API_KEY is not set; sentiment will be blank")


# -----------------------------------------------------------------------------
# Google Sheets helpers
# -----------------------------------------------------------------------------
def get_gspread_client() -> gspread.Client:
    creds_info = json.loads(GOOGLE_CREDS_JSON)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    return gspread.authorize(creds)


def read_pairs_from_sheet(sheet_name: str, tab_name: str) -> List[str]:
    """Read column A (pairs) from the screener sheet, skipping header and blanks."""
    gc = get_gspread_client()
    sh = gc.open(sheet_name)
    ws = sh.worksheet(tab_name)

    col_a = ws.col_values(1)  # 1-based index
    if not col_a or len(col_a) <= 1:
        return []

    # Skip header (row 1)
    pairs = [v.strip() for v in col_a[1:] if v.strip()]
    return pairs


def clean_value(v: Any) -> Any:
    """Make values safe for Sheets (no NaN / None)."""
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return v


def update_sentiment_columns(
    sheet_name: str, tab_name: str, sentiments: List[Dict[str, Any]]
) -> None:
    """
    Write sentiment data into columns T–Z.
    T: news_score_7d
    U: news_label_7d
    V: news_bullish_ratio_7d
    W: news_bearish_ratio_7d
    X: news_articles_7d
    Y: news_top_keywords_7d
    Z: news_emoji_7d
    """
    if not sentiments:
        logger.info("No sentiments to write")
        return

    gc = get_gspread_client()
    sh = gc.open(sheet_name)
    ws = sh.worksheet(tab_name)

    headers = [
        [
            "news_score_7d",
            "news_label_7d",
            "news_bullish_ratio_7d",
            "news_bearish_ratio_7d",
            "news_articles_7d",
            "news_top_keywords_7d",
            "news_emoji_7d",
        ]
    ]
    ws.update(values=headers, range_name="T1:Z1")

    values: List[List[Any]] = []
    for s in sentiments:
        row = [
            clean_value(s.get("news_score_7d")),
            clean_value(s.get("news_label_7d")),
            clean_value(s.get("news_bullish_ratio_7d")),
            clean_value(s.get("news_bearish_ratio_7d")),
            clean_value(s.get("news_articles_7d")),
            clean_value(s.get("news_top_keywords_7d")),
            clean_value(s.get("news_emoji_7d")),
        ]
        values.append(row)

    last_row = len(values) + 1  # +1 for header
    range_name = f"T2:Z{last_row}"
    logger.info("Updating sentiment columns %s", range_name)
    ws.update(values=values, range_name=range_name)


# -----------------------------------------------------------------------------
# ForexNewsAPI sentiment
# -----------------------------------------------------------------------------
def score_to_emoji(score: Any) -> str:
    """Map numeric sentiment score to a simple emoji."""
    if not isinstance(score, (int, float)):
        return ""
    if score >= 0.5:
        return "🟢"  # strong bullish
    if score >= 0.2:
        return "🟡"  # mildly bullish / positive
    if score <= -0.5:
        return "🔴"  # strong bearish
    if score <= -0.2:
        return "🟠"  # mildly bearish / negative
    return "⚪"  # neutral / mixed


def fetch_sentiment_for_pair(session: requests.Session, pair: str) -> Dict[str, Any]:
    """
    Call ForexNewsAPI /stat endpoint for a single FX pair and return
    a sentiment metrics dict. On error, log details and return neutral/NaN values.
    """
    api_key = FOREX_NEWS_API_KEY
    if not api_key:
        logger.error("FOREX_NEWS_API_KEY is not set; skipping sentiment for %s", pair)
        return {
            "news_score_7d": float("nan"),
            "news_label_7d": "",
            "news_bullish_ratio_7d": float("nan"),
            "news_bearish_ratio_7d": float("nan"),
            "news_articles_7d": 0,
            "news_top_keywords_7d": "",
            "news_emoji_7d": "",
        }

    # Oanda uses "EUR_USD", ForexNewsAPI wants "EUR-USD"
    cp = pair.replace("_", "-")

    url = FOREX_NEWS_SENTIMENT_URL.rstrip("/")  # should be .../api/v1/stat
    params = {
        "currencypair": cp,
        "date": FOREX_NEWS_DATE,  # e.g. "last7days" or "last30days"
        "page": 1,
        "token": api_key,
    }

    try:
        resp = session.get(url, params=params, timeout=15)
        text_preview = resp.text[:500] if resp.text else ""

        if resp.status_code != 200:
            logger.warning(
                "HTTP error (sentiment) for pair %s (%s): %s | body: %s",
                pair,
                cp,
                resp.status_code,
                text_preview,
            )
            return {
                "news_score_7d": float("nan"),
                "news_label_7d": "",
                "news_bullish_ratio_7d": float("nan"),
                "news_bearish_ratio_7d": float("nan"),
                "news_articles_7d": 0,
                "news_top_keywords_7d": "",
                "news_emoji_7d": "",
            }

        data = resp.json()
        # If API wraps stats in "data", unwrap that
        if isinstance(data, dict) and "data" in data:
            root = data.get("data", {})
        else:
            root = data

    except Exception as e:
        logger.warning("Exception calling sentiment API for %s (%s): %s", pair, cp, e)
        return {
            "news_score_7d": float("nan"),
            "news_label_7d": "",
            "news_bullish_ratio_7d": float("nan"),
            "news_bearish_ratio_7d": float("nan"),
            "news_articles_7d": 0,
            "news_top_keywords_7d": "",
            "news_emoji_7d": "",
        }

    # -------------------------------------------------------------------------
    # Best-effort field extraction (we'll see exact keys from logs)
    # -------------------------------------------------------------------------
    score = None
    label = ""
    bull = None
    bear = None
    news_count: Any = None
    top_keywords: List[str] = []

    if isinstance(root, dict):
        score = (
            root.get("sentiment_avg")
            or root.get("sentiment_score")
            or root.get("sentiment")
            or root.get("score")
        )

        label = (
            root.get("sentiment_label")
            or root.get("label")
            or root.get("overall_sentiment")
            or ""
        )

        bull = root.get("bullish_ratio") or root.get("bullish_percent")
        bear = root.get("bearish_ratio") or root.get("bearish_percent")

        # Normalize percentages > 1 into [0,1]
        if isinstance(bull, (int, float)) and bull > 1.0:
            bull = bull / 100.0
        if isinstance(bear, (int, float)) and bear > 1.0:
            bear = bear / 100.0

        news_count = (
            root.get("news_count")
            or root.get("total")
            or (len(root.get("sentiments", [])) if isinstance(root.get("sentiments"), list) else None)
        )

        kws = root.get("top_keywords") or root.get("keywords") or []
        if isinstance(kws, list):
            top_keywords = [str(k) for k in kws]
        elif isinstance(kws, str):
            top_keywords = [kws]

    emoji = score_to_emoji(score)

    return {
        "news_score_7d": score,
        "news_label_7d": label,
        "news_bullish_ratio_7d": bull,
        "news_bearish_ratio_7d": bear,
        "news_articles_7d": news_count or 0,
        "news_top_keywords_7d": ", ".join(top_keywords) if top_keywords else "",
        "news_emoji_7d": emoji,
    }


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def run_sentiment_once() -> None:
    pairs = read_pairs_from_sheet(GOOGLE_SHEET_NAME, GOOGLE_SHEET_TAB)
    if not pairs:
        logger.warning("No pairs found in sheet %s tab %s", GOOGLE_SHEET_NAME, GOOGLE_SHEET_TAB)
        return

    logger.info(
        "Updating news sentiment (Sentiment endpoint) for %d pairs", len(pairs)
    )

    sentiments: List[Dict[str, Any]] = []
    with requests.Session() as session:
        for idx, pair in enumerate(pairs, start=1):
            logger.info("(%d/%d) Fetching sentiment for %s", idx, len(pairs), pair)
            s = fetch_sentiment_for_pair(session, pair)
            sentiments.append(s)

    update_sentiment_columns(GOOGLE_SHEET_NAME, GOOGLE_SHEET_TAB, sentiments)


def main() -> None:
    logger.info(
        "Starting Forex sentiment bot (Sentiment endpoint). Sheet='%s' Tab='%s' Interval=%ss",
        GOOGLE_SHEET_NAME,
        GOOGLE_SHEET_TAB,
        FOREX_SENTIMENT_INTERVAL,
    )
    while True:
        try:
            run_sentiment_once()
        except Exception as e:
            logger.exception("Error in sentiment loop: %s", e)
        logger.info("Sleeping for %s seconds...", FOREX_SENTIMENT_INTERVAL)
        time.sleep(FOREX_SENTIMENT_INTERVAL)


if __name__ == "__main__":
    main()
