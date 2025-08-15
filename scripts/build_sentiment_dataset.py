import argparse
import os
import sys
import csv
from typing import List

SCHEMA = ["date", "ticker", "text", "label"]

SAMPLES = [
	("2025-02-01", "EURUSD=X", "Euro rallies on strong PMI", 2),
	("2025-02-02", "EURUSD=X", "Euro dips after weak retail sales", 0),
	("2025-02-03", "EURUSD=X", "Euro stable amid mixed signals", 1),
	("2025-02-04", "EURUSD=X", "ECB hints at possible rate hike", 2),
	("2025-02-05", "EURUSD=X", "Uncertainty over GDP weighs on euro", 0),
]


def write_csv(path: str, rows: List[tuple]):
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	with open(path, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(SCHEMA)
		for r in rows:
			w.writerow(r)


def main(argv: List[str]) -> int:
	parser = argparse.ArgumentParser(description="Build a minimal sentiment dataset (date,ticker,text,label)")
	parser.add_argument("--output", type=str, default="data/processed/sentiment.csv")
	parser.add_argument("--ticker", type=str, default="EURUSD=X")
	args = parser.parse_args(argv)

	rows = [(d, args.ticker, t, l) for (d, _, t, l) in SAMPLES]
	write_csv(args.output, rows)
	print(f"[sentiment-builder] Wrote {args.output} with {len(rows)} rows")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))