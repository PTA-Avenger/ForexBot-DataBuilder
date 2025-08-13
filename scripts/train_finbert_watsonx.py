import argparse
import sys
import time
from typing import List


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="FinBERT fine-tuning entrypoint (Watsonx.ai)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--dataset", type=str, default="data/processed/sentiment.csv")
    args = parser.parse_args(argv)

    print(
        f"[train_finbert] Starting fine-tuning with epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.learning_rate}"
    )
    print(f"[train_finbert] Dataset: {args.dataset}")
    for epoch in range(1, args.epochs + 1):
        time.sleep(0.1)
        print(f"[train_finbert] Epoch {epoch}/{args.epochs} - loss=... val_loss=...")
    print("[train_finbert] Fine-tuning complete. Saving model artifact to outputs/... (simulated)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
