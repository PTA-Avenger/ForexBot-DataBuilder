import argparse
import os
import sys
from typing import List

import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	return df


def build_features(df: pd.DataFrame):
	y = torch.tensor(df["target"].astype(int).values, dtype=torch.long)
	feature_cols = [c for c in df.columns if c not in {"date", "ticker", "target"}]
	X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
	return X, y


class MLPClassifier(nn.Module):
	def __init__(self, input_dim: int):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 2),
		)

	def forward(self, x):
		return self.net(x)


def train_model(X: torch.Tensor, y: torch.Tensor, epochs: int, batch_size: int, lr: float) -> dict:
	X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42, stratify=y.numpy())
	X_train = torch.tensor(X_train, dtype=torch.float32)
	y_train = torch.tensor(y_train, dtype=torch.long)
	X_test = torch.tensor(X_test, dtype=torch.float32)
	y_test = torch.tensor(y_test, dtype=torch.long)

	model = MLPClassifier(X_train.shape[1]).to(DEVICE)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss()

	model.train()
	for epoch in range(1, epochs + 1):
		perm = torch.randperm(X_train.size(0))
		for i in range(0, X_train.size(0), batch_size):
			idx = perm[i:i + batch_size]
			inputs = X_train[idx].to(DEVICE)
			targets = y_train[idx].to(DEVICE)
			optimizer.zero_grad()
			logits = model(inputs)
			loss = criterion(logits, targets)
			loss.backward()
			optimizer.step()

	model.eval()
	with torch.no_grad():
		preds = model(X_test.to(DEVICE)).argmax(dim=1).cpu()
		report = classification_report(y_test, preds, output_dict=True)

	os.makedirs(OUTPUT_DIR, exist_ok=True)
	model_path = os.path.join(OUTPUT_DIR, "lstm_model.pt")
	torch.save(model.state_dict(), model_path)
	metrics_path = os.path.join(OUTPUT_DIR, "lstm_metrics.json")
	pd.Series(report).to_json(metrics_path)
	return {"model_path": model_path, "metrics_path": metrics_path}


def main(argv: List[str]) -> int:
	parser = argparse.ArgumentParser(description="LSTM training entrypoint (Watsonx.ai)")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--dataset", type=str, default="data/processed/trend_dataset.csv")
	args = parser.parse_args(argv)

	print(f"[train_lstm] Starting training with epochs={args.epochs} batch={args.batch_size} lr={args.learning_rate}")
	print(f"[train_lstm] Dataset: {args.dataset}")
	df = load_dataset(args.dataset)
	X, y = build_features(df)
	artifacts = train_model(X, y, args.epochs, args.batch_size, args.learning_rate)
	print(f"[train_lstm] Saved model to {artifacts['model_path']}")
	print(f"[train_lstm] Saved metrics to {artifacts['metrics_path']}")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))
