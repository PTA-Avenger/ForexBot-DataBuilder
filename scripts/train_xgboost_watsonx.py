import argparse
import sys
import time
from typing import List


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="XGBoost training entrypoint (Watsonx.ai)")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--dataset", type=str, default="data/processed/meanrev_dataset.csv")
    args = parser.parse_args(argv)

    print(
        f"[train_xgboost] Starting training with n_estimators={args.n_estimators} "
        f"max_depth={args.max_depth} lr={args.learning_rate}"
    )
    print(f"[train_xgboost] Dataset: {args.dataset}")
    for step in range(1, 6):
        time.sleep(0.1)
        print(f"[train_xgboost] Step {step}/5 - train_auc=... val_auc=...")
    print("[train_xgboost] Training complete. Saving model artifact to outputs/... (simulated)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
