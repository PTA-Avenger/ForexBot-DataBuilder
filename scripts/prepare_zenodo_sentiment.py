import argparse
import os
import sys
from typing import List, Optional

import pandas as pd

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def normalize_label(val: Optional[object]) -> Optional[int]:
	if val is None:
		return None
	if isinstance(val, str):
		vs = val.strip().lower()
		if vs in LABEL_MAP:
			return LABEL_MAP[vs]
		# numeric strings: try int
		try:
			num = int(vs)
		except Exception:
			num = None
		if num is not None:
			if num in (0, 1, 2):
				return num
			if num in (-1, 0, 1):
				return num + 1
			return None
		return None
	if isinstance(val, (int,)):
		if val in (0, 1, 2):
			return val
		if val in (-1, 0, 1):
			return val + 1
		return None
	return None


def main(argv: List[str]) -> int:
	parser = argparse.ArgumentParser(description="Convert Zenodo sentiment CSV to data/processed/sentiment.csv schema")
	parser.add_argument("--input", type=str, required=True, help="Path to raw Zenodo CSV")
	parser.add_argument("--output", type=str, default="data/processed/sentiment.csv")
	args = parser.parse_args(argv)

	if not os.path.exists(args.input):
		print(f"[zenodo-convert] Input not found: {args.input}", file=sys.stderr)
		return 2

	df = pd.read_csv(args.input)
	required_cols = {"published_at", "ticker", "text"}
	missing = required_cols - set(df.columns)
	if missing:
		print(f"[zenodo-convert] Missing required columns in input: {sorted(missing)}", file=sys.stderr)
		return 3

	label_col = None
	for c in ["true_sentiment", "finbert_sentiment", "label"]:
		if c in df.columns:
			label_col = c
			break
	if label_col is None:
		print("[zenodo-convert] Could not find a sentiment column (true_sentiment/finbert_sentiment/label)", file=sys.stderr)
		return 4

	labels = df[label_col].apply(normalize_label)
	texts = df["text"].fillna("").astype(str)
	fallback_titles = df["title"].fillna("").astype(str) if "title" in df.columns else ""
	texts = texts.where(texts.str.len() > 0, other=fallback_titles)

	out = pd.DataFrame(
		{
			"date": pd.to_datetime(df["published_at"], errors="coerce").dt.date.astype(str),
			"ticker": df["ticker"].astype(str),
			"text": texts,
			"label": labels,
		}
	)
	before = len(out)
	out = out.dropna(subset=["date", "ticker", "text", "label"]).copy()
	out = out[out["text"].str.len() > 0]
	out = out[out["label"].isin([0, 1, 2])]
	after = len(out)
	os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
	out.to_csv(args.output, index=False)
	print(f"[zenodo-convert] Wrote {args.output} with {after} rows (dropped {before - after})")
	print("[zenodo-convert] Label counts:\n" + out["label"].value_counts().sort_index().to_string())
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))