import argparse
import os
import sys
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")


def load_dataset(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
	# Expect a tabular dataset with target column named 'target'
	target = df["target"].astype(int)
	# Drop non-feature columns
	feature_cols = [
		c for c in df.columns
		if c not in {"date", "ticker", "target"}
	]
	X = df[feature_cols]
	return X, target


def train_and_eval(X: pd.DataFrame, y: pd.Series, n_estimators: int, max_depth: int, learning_rate: float) -> dict:
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
	clf = xgb.XGBClassifier(
		n_estimators=n_estimators,
		max_depth=max_depth,
		learning_rate=learning_rate,
		subsample=0.9,
		colsample_bytree=0.9,
		n_jobs=4,
		tree_method="hist",
	)
	clf.fit(X_train, y_train)
	preds = clf.predict(X_test)
	report = classification_report(y_test, preds, output_dict=True)
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	model_path = os.path.join(OUTPUT_DIR, "xgboost_model.json")
	clf.save_model(model_path)
	metrics_path = os.path.join(OUTPUT_DIR, "xgboost_metrics.json")
	pd.Series(report).to_json(metrics_path)
	return {"model_path": model_path, "metrics_path": metrics_path}


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
	df = load_dataset(args.dataset)
	X, y = build_features(df)
	artifacts = train_and_eval(X, y, args.n_estimators, args.max_depth, args.learning_rate)
	print(f"[train_xgboost] Saved model to {artifacts['model_path']}")
	print(f"[train_xgboost] Saved metrics to {artifacts['metrics_path']}")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))
