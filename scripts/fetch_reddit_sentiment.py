import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import praw  # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

MODEL_NAME = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")
DEFAULT_SUBREDDITS = ["Forex", "FinancialMarkets", "economics", "ForexTrading"]
DEFAULT_TICKERS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]

# Map detected sentiment string to numeric label used by our trainers
FINBERT_TO_LABEL: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}

# Simple heuristics to map text to a currency pair ticker
TICKER_SYNONYMS: Dict[str, List[str]] = {
    "EURUSD=X": ["eurusd", "eur/usd", "eur usd", "eur usd pair", "euro dollar"],
    "GBPUSD=X": ["gbpusd", "gbp/usd", "gbp usd", "cable"],
    "USDJPY=X": ["usdjpy", "usd/jpy", "usd jpy"],
    "AUDUSD=X": ["audusd", "aud/usd", "aud usd"],
}


def init_reddit() -> praw.Reddit:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "forexbot-databuilder/0.1")
    if not client_id or not client_secret:
        raise EnvironmentError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET (and optionally REDDIT_USER_AGENT)")
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    reddit.read_only = True
    return reddit


def init_pipeline() -> TextClassificationPipeline:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, task="sentiment-analysis")


def detect_ticker(text: str, allowed_tickers: List[str], default_ticker: Optional[str]) -> Optional[str]:
    text_l = text.lower()
    for ticker in allowed_tickers:
        for syn in TICKER_SYNONYMS.get(ticker, []):
            if syn in text_l:
                return ticker
    return default_ticker


def to_iso_date(utc_ts: float) -> str:
    return datetime.fromtimestamp(utc_ts, tz=timezone.utc).date().isoformat()


def write_rows(csv_path: str, rows: List[Tuple[str, str, str, int]], append: bool) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    mode = "a" if append and os.path.exists(csv_path) else "w"
    with open(csv_path, mode, newline="") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["date", "ticker", "text", "label"])  # header
        for r in rows:
            w.writerow(r)


def fetch_posts(
    reddit: praw.Reddit,
    subreddits: List[str],
    query: Optional[str],
    days: int,
    limit: int,
    min_score: int,
) -> List[praw.models.Submission]:
    results: List[praw.models.Submission] = []
    earliest_ts = (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp()
    for sub in subreddits:
        sr = reddit.subreddit(sub)
        try:
            if query:
                # time_filter granularity is coarse; we filter by created_utc ourselves
                for post in sr.search(query=query, sort="new", time_filter="week", limit=limit):
                    if post.created_utc >= earliest_ts and post.score >= min_score:
                        results.append(post)
            else:
                for post in sr.new(limit=limit):
                    if post.created_utc >= earliest_ts and post.score >= min_score:
                        results.append(post)
        except Exception as exc:  # pragma: no cover
            print(f"[reddit] Warning: failed to fetch from r/{sub}: {exc}")
        time.sleep(0.5)  # be gentle
    # Deduplicate by id
    unique = {p.id: p for p in results}
    return list(unique.values())


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Fetch Reddit posts and build sentiment.csv labeled by FinBERT")
    parser.add_argument("--subreddits", type=str, default=",".join(DEFAULT_SUBREDDITS))
    parser.add_argument("--query", type=str, default="EURUSD OR GBPUSD OR USDJPY OR AUDUSD")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-score", type=int, default=1)
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--default-ticker", type=str, default="EURUSD=X")
    parser.add_argument("--output", type=str, default="data/processed/sentiment.csv")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args(argv)

    reddit = init_reddit()
    classifier = init_pipeline()

    subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    posts = fetch_posts(
        reddit=reddit,
        subreddits=subreddits,
        query=args.query,
        days=args.days,
        limit=args.limit,
        min_score=args.min_score,
    )

    rows: List[Tuple[str, str, str, int]] = []
    texts: List[str] = []
    meta: List[Tuple[str, Optional[str]]] = []  # (date, ticker)

    for p in posts:
        text = f"{p.title}\n{getattr(p, 'selftext', '') or ''}".strip()
        if not text:
            continue
        ticker = detect_ticker(text, allowed_tickers=tickers, default_ticker=args.default_ticker)
        if ticker is None:
            continue
        texts.append(text)
        meta.append((to_iso_date(p.created_utc), ticker))

    if not texts:
        print("[reddit] No eligible posts found.")
        return 0

    # Batch classify for speed
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_meta = meta[i : i + batch_size]
        outputs = classifier(batch_texts)
        for (date_iso, ticker), text, out in zip(batch_meta, batch_texts, outputs):
            label_str = out["label"].lower()
            label = FINBERT_TO_LABEL.get(label_str)
            if label is None:
                continue
            rows.append((date_iso, ticker, text, label))

    if not rows:
        print("[reddit] No labeled rows to write.")
        return 0

    write_rows(args.output, rows, append=args.append)
    print(f"[reddit] Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))