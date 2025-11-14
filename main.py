import os
import json
import time
import logging
import re
from datetime import datetime, timedelta, timezone
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ---------- Logging setup ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------- Env & constants ----------

FOREX_NEWS_API_KEY = os.getenv("FOREX_NEWS_API_KEY")

# Sheet / tab names – support multiple env var names, with sensible defaults
SHEET_NAME = (
    os.getenv("FOREX_SHEET_NAME")
    or os.getenv("GOOGLE_SHEET_NAME")
    or "Active-Investing"
)
WORKSHEET_NAME = (
    os.getenv("FOREX_SHEET_TAB")
    or os.getenv("GOOGLE_WORKSHEET_NAME")
    or "Oanda-Screener"
)

# How often to run
UPDATE_INTERVAL_SECONDS = int(os.getenv("FOREX_SENTIMENT_INTERVAL_SECONDS", "21600"))

# How many days of news to consider (we enforce this *ourselves*, not via API params)
WINDOW_DAYS = int(os.getenv("FOREX_SENTIMENT_WINDOW_DAYS", "7"))

# Max news items per pair per API call
MAX_ITEMS_PER_PAIR = int(os.getenv("FOREX_NEWS_MAX_ITEMS_PER_PAIR", "50"))

# ---------- Helpers ----------


def column_index_to_letter(idx: int) -> str:
    """Convert 1-based column index to Excel-style letter (1 -> A, 27 -> AA)."""
    letters = ""
    while idx > 0:
        idx, remainder = divmod(idx - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def normalize_header_name(name: str) -> str:
    """Lowercase, strip, and collapse non-alphanumerics to underscores."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def parse_article_date(date_str: str) -> Optional[datetime]:
    """Parse ISO-like date from API. Returns timezone-aware UTC datetime or None."""
    if not date_str:
        return None

    # Common formats, e.g. 2025-11-14T03:45:00-04:00 or with Z
    try:
        # Normalize Z to +00:00 for fromisoformat
        if date_str.endswith("Z"):
            date_str = date_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            # Assume UTC if no tz given
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


# ---------- ForexNews API client ----------


class ForexNewsAPIClient:
    BASE_URL = "https://forexnewsapi.com/api/v1"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("FOREX_NEWS_API_KEY is not set")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ForexSentimentBot/1.0"})

    def get_pair_news(self, oanda_pair: str) -> Dict[str, Any]:
        """
        Call the *news* endpoint (NOT /stat) for a single currency pair.

        We DO NOT send &date=... because your plan doesn’t allow it.
        Instead, we filter by date locally.
        """
        currencypair = oanda_pair.replace("_", "-").upper().strip()

        params = {
            "currencypair": currencypair,
            "items": MAX_ITEMS_PER_PAIR,
            "token": self.api_key,
            # do NOT include 'date' – plan does not allow it
        }

        resp = self.session.get(self.BASE_URL, params=params, timeout=20)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            body = None
            try:
                body = resp.text
            except Exception:
                pass
            msg = f"{resp.status_code}"
            if body:
                msg += f" | body: {body}"
            logging.warning("HTTP error (news) for pair %s: %s", currencypair, msg)
            raise

        try:
            data = resp.json()
        except ValueError:
            logging.warning(
                "Non-JSON response from ForexNewsAPI for %s: %s",
                currencypair,
                resp.text[:300],
            )
            raise

        # Some of these APIs use {"data": [...]} and some just return a list
        if isinstance(data, dict) and "data" in data:
            articles = data.get("data") or []
        elif isinstance(data, list):
            articles = data
        else:
            logging.warning(
                "Unexpected JSON structure for %s: %s",
                currencypair,
                str(data)[:300],
            )
            articles = []

        return {
            "currencypair": currencypair,
            "raw": data,
            "articles": articles,
        }

    def summarize_pair_sentiment(
        self, oanda_pair: str, window_days: int = WINDOW_DAYS
    ) -> Dict[str, Any]:
        """
        High-level helper: fetch news for a pair and compute sentiment summary.

        Returns dict with keys like:
        - status, error
        - article_count, positive_count, negative_count, neutral_count
        - avg_sentiment_score
        - sentiment_label (bullish/bearish/neutral)
        - emoji
        - top_keywords
        - last_updated
        """
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(days=window_days)

        try:
            payload = self.get_pair_news(oanda_pair)
        except Exception as e:
            # Any HTTP/JSON problems become a clean error result
            return {
                "status": "error",
                "error": str(e),
                "article_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "avg_sentiment_score": None,
                "sentiment_label": "error",
                "emoji": "❌",
                "top_keywords": "error",
                "last_updated": now_utc.isoformat(),
            }

        articles = payload.get("articles", [])
        # Filter articles to last N days based on their 'date' field
        recent_articles: List[Dict[str, Any]] = []
        for art in articles:
            date_str = art.get("date") or art.get("published_at")
            dt = parse_article_date(date_str) if date_str else None
            if dt is None:
                # If we can't parse the date, include it (being generous)
                recent_articles.append(art)
            elif dt >= cutoff:
                recent_articles.append(art)

        if not recent_articles:
            return {
                "status": "no_data",
                "error": None,
                "article_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "avg_sentiment_score": None,
                "sentiment_label": "none",
                "emoji": "➖",
                "top_keywords": "",
                "last_updated": now_utc.isoformat(),
            }

        pos = neg = neu = 0
        score_sum = 0.0
        score_count = 0

        keyword_counter: Counter = Counter()

        for art in recent_articles:
            sentiment = (art.get("sentiment") or "").lower()
            score = art.get("sentiment_score")

            if sentiment == "positive":
                pos += 1
            elif sentiment == "negative":
                neg += 1
            else:
                neu += 1

            if isinstance(score, (int, float)):
                score_sum += float(score)
                score_count += 1

            # Collect simple keywords from title
            title = art.get("title") or ""
            words = re.findall(r"[A-Za-z]{4,}", title)
            for w in words:
                wl = w.lower()
                # basic stopwords
                if wl in {"with", "from", "that", "this", "will", "have", "been"}:
                    continue
                keyword_counter[wl] += 1

        avg_score = score_sum / score_count if score_count > 0 else None

        # Decide overall sentiment label
        if pos > neg and pos >= neu:
            label = "bullish"
            emoji = "🟢"
        elif neg > pos and neg >= neu:
            label = "bearish"
            emoji = "🔴"
        else:
            label = "neutral"
            emoji = "⚪"

        top_keywords = ", ".join(
            [kw for kw, _ in keyword_counter.most_common(5)]
        )

        return {
            "status": "ok",
            "error": None,
            "article_count": len(recent_articles),
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "avg_sentiment_score": avg_score,
            "sentiment_label": label,
            "emoji": emoji,
            "top_keywords": top_keywords,
            "last_updated": now_utc.isoformat(),
        }


# ---------- Google Sheets helpers ----------


def get_gspread_client():
    creds_json = os.getenv("GOOGLE_CREDS_JSON")
    if not creds_json:
        raise ValueError("GOOGLE_CREDS_JSON env var is not set")

    try:
        creds_dict = json.loads(creds_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"GOOGLE_CREDS_JSON is not valid JSON: {e}")

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client


def get_worksheet():
    client = get_gspread_client()
    sheet = client.open(SHEET_NAME)
    ws = sheet.worksheet(WORKSHEET_NAME)
    return ws


def find_pair_column_index(header_row: List[str]) -> int:
    """
    Try to find the column that contains the Oanda pair codes.
    We look for headers containing 'pair', 'instrument', or 'symbol'.
    If none, default to column 1.
    """
    for idx, name in enumerate(header_row, start=1):
        norm = normalize_header_name(name)
        if any(key in norm for key in ("pair", "instrument", "symbol")):
            return idx
    return 1  # fallback: first column


def get_pairs_from_sheet(ws, max_pairs: Optional[int] = None) -> List[str]:
    header = ws.row_values(1)
    pair_col_idx = find_pair_column_index(header)
    col_vals = ws.col_values(pair_col_idx)[1:]  # skip header

    pairs: List[str] = [v.strip() for v in col_vals if v.strip()]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def find_sentiment_block(ws) -> Tuple[int, int, List[str]]:
    """
    Find contiguous 'news_*' columns in the header row.
    Returns (start_idx, end_idx, header_names).
    """
    header = ws.row_values(1)
    indices = [i for i, name in enumerate(header, start=1)
               if normalize_header_name(name).startswith("news_")]

    if not indices:
        raise RuntimeError("No 'news_*' columns found in header row")

    start_idx = min(indices)
    end_idx = max(indices)
    block_headers = header[start_idx - 1:end_idx]
    return start_idx, end_idx, block_headers


def build_row_values_for_summary(
    block_headers: List[str], summary: Dict[str, Any]
) -> List[Any]:
    """
    Map our sentiment summary dict into a row of values aligned with
    the 'news_*' header names in block_headers.
    """
    values: List[Any] = []

    for name in block_headers:
        norm = normalize_header_name(name)

        if norm == "news_sentiment_7d":
            values.append(summary.get("sentiment_label"))
        elif norm == "news_score_7d":
            values.append(summary.get("avg_sentiment_score"))
        elif norm == "news_count_7d":
            values.append(summary.get("article_count"))
        elif norm == "news_positive_7d":
            values.append(summary.get("positive_count"))
        elif norm == "news_negative_7d":
            values.append(summary.get("negative_count"))
        elif norm == "news_neutral_7d":
            values.append(summary.get("neutral_count"))
        elif norm == "news_top_keywords_7d":
            values.append(summary.get("top_keywords"))
        elif norm == "news_emoji_7d":
            values.append(summary.get("emoji"))
        elif norm == "news_last_updated_7d":
            values.append(summary.get("last_updated"))
        else:
            # Unknown news_* column – leave blank so we don't stomp on anything
            values.append("")
    return values


# ---------- Main update logic ----------


def update_sentiment_once():
    logging.info(
        "Starting Forex sentiment bot (Sentiment endpoint). Sheet='%s' Tab='%s' Interval=%ss",
        SHEET_NAME,
        WORKSHEET_NAME,
        UPDATE_INTERVAL_SECONDS,
    )

    ws = get_worksheet()
    header = ws.row_values(1)

    # Get pairs
    pairs = get_pairs_from_sheet(ws)
    pair_count = len(pairs)
    logging.info("Updating news sentiment (Sentiment endpoint) for %d pairs", pair_count)

    # Find the block of news_* columns to update
    start_col_idx, end_col_idx, block_headers = find_sentiment_block(ws)
    start_row = 2
    end_row = start_row + pair_count - 1

    start_col_letter = column_index_to_letter(start_col_idx)
    end_col_letter = column_index_to_letter(end_col_idx)
    range_a1 = f"{start_col_letter}{start_row}:{end_col_letter}{end_row}"

    logging.info("Will update sentiment columns %s", range_a1)

    fx_client = ForexNewsAPIClient(FOREX_NEWS_API_KEY)

    all_rows: List[List[Any]] = []

    for i, pair in enumerate(pairs, start=1):
        logging.info("(%d/%d) Fetching sentiment for %s", i, pair_count, pair)
        summary = fx_client.summarize_pair_sentiment(pair, window_days=WINDOW_DAYS)

        # If there was a hard API error (403, etc.), mark row as "error" but still
        # keep going so at least we can see something in the sheet.
        if summary.get("status") == "error":
            # overwrite some fields to make it obvious in the sheet
            summary["sentiment_label"] = "error"
            summary["emoji"] = "❌"
            if not summary.get("top_keywords"):
                summary["top_keywords"] = "error"

        row_values = build_row_values_for_summary(block_headers, summary)
        all_rows.append(row_values)

    logging.info("Updating sentiment columns %s", range_a1)
    ws.update(range_a1, all_rows, value_input_option="USER_ENTERED")
    logging.info("Done updating sheet.")


def main_loop():
    while True:
        try:
            update_sentiment_once()
        except Exception as e:
            logging.exception("Unexpected error in update loop: %s", e)
            # Best-effort: write a single 'error' marker in the first row of the block
            try:
                ws = get_worksheet()
                start_col_idx, end_col_idx, block_headers = find_sentiment_block(ws)
                start_col_letter = column_index_to_letter(start_col_idx)
                # Put timestamp + 'error' in the last two news_* columns if they exist
                now_iso = datetime.now(timezone.utc).isoformat()
                error_row = ["" for _ in block_headers]
                # Try to find keywords / emoji columns if present
                for idx, name in enumerate(block_headers):
                    norm = normalize_header_name(name)
                    if norm == "news_top_keywords_7d":
                        error_row[idx] = now_iso
                    elif norm == "news_emoji_7d":
                        error_row[idx] = "error"
                ws.update(
                    f"{start_col_letter}2:{column_index_to_letter(start_col_idx + len(block_headers) - 1)}2",
                    [error_row],
                    value_input_option="USER_ENTERED",
                )
            except Exception:
                # Don't let error-reporting crash loop
                logging.exception("Failed to write error marker to sheet")

        logging.info("Sleeping for %d seconds...", UPDATE_INTERVAL_SECONDS)
        time.sleep(UPDATE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main_loop()
