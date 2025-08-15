import argparse
import os
import sys
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")
MODEL_NAME = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	return df


def prepare_data(df: pd.DataFrame):
	# Expect columns: text, label
	if df.empty or len(df) < 2:
		return None, None
	if "text" not in df.columns or "label" not in df.columns:
		raise ValueError("sentiment.csv must have columns: text,label")
	X_train, X_test, y_train, y_test = train_test_split(df["text"].tolist(), df["label"].astype(int).tolist(), test_size=0.2, random_state=42, stratify=df["label"].astype(int).tolist())
	return (X_train, y_train), (X_test, y_test)


class TextDataset(torch.utils.data.Dataset):
	def __init__(self, texts, labels, tokenizer):
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx):
		enc = self.tokenizer(
			self.texts[idx],
			truncation=True,
			padding="max_length",
			max_length=128,
			return_tensors="pt",
		)
		item = {k: v.squeeze(0) for k, v in enc.items()}
		item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
		return item


def train_and_eval(X_train, y_train, X_test, y_test) -> dict:
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(set(y_train)))

	train_ds = TextDataset(X_train, y_train, tokenizer)
	test_ds = TextDataset(X_test, y_test, tokenizer)

	os.makedirs(OUTPUT_DIR, exist_ok=True)
	args = TrainingArguments(
		output_dir=OUTPUT_DIR,
		num_train_epochs=1,
		per_device_train_batch_size=8,
		per_device_eval_batch_size=8,
		evaluation_strategy="epoch",
		logging_steps=10,
		save_strategy="no",
		report_to=[],
	)
	trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
	trainer.train()
	metrics = trainer.evaluate()
	metrics_path = os.path.join(OUTPUT_DIR, "finbert_metrics.json")
	pd.Series(metrics).to_json(metrics_path)
	model_path = os.path.join(OUTPUT_DIR, "finbert_model")
	model.save_pretrained(model_path)
	tokenizer.save_pretrained(model_path)
	return {"model_path": model_path, "metrics_path": metrics_path}


def main(argv: List[str]) -> int:
	parser = argparse.ArgumentParser(description="FinBERT fine-tuning entrypoint (Watsonx.ai)")
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--learning-rate", type=float, default=2e-5)
	parser.add_argument("--dataset", type=str, default="data/processed/sentiment.csv")
	args = parser.parse_args(argv)

	print(
		f"[train_finbert] Starting fine-tuning with epochs={args.epochs} "
		f"batch={args.batch_size} lr={args.learning_rate}"
	)
	print(f"[train_finbert] Dataset: {args.dataset}")
	df = load_dataset(args.dataset)
	train_data, test_data = prepare_data(df)
	if train_data is None:
		print("[train_finbert] Dataset is empty or insufficient; skipping training.")
		return 0
	artifacts = train_and_eval(train_data[0], train_data[1], test_data[0], test_data[1])
	print(f"[train_finbert] Saved model to {artifacts['model_path']}")
	print(f"[train_finbert] Saved metrics to {artifacts['metrics_path']}")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))
