"""Ensemble evaluation: average predictions from multiple trained models.

Usage::

    python ensemble_evaluate.py --model_dirs model_a/ model_b/ model_c/ --data_dir ./data --fold 0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

COLUMNS_TO_REMOVE: list[str] = ["source"]


def create_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Ensemble evaluation over multiple saved models.")
    parser.add_argument(
        "--model_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to saved model directories.",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Root data directory (from prepare_datasets.py).")
    parser.add_argument("--fold", type=int, default=None, help="Fold index (0-9). Requires --data_dir.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "dev"],
        help="Dataset split to evaluate on.",
    )
    return parser


def get_predictions(model_dir: str, raw_dataset: DatasetDict, split: str) -> np.ndarray:
    """Load a model, tokenize with its own tokenizer, and return predictions.

    Args:
        model_dir: Path to a saved model directory.
        raw_dataset: Non-tokenized HF DatasetDict (needs 'original', 'simplification', 'label').
        split: Split name to evaluate on ('test' or 'dev').

    Returns:
        1-D array of predicted scores, NaN/Inf replaced with 0.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example: dict) -> dict:
        return tokenizer(example["original"], example["simplification"], truncation=True, padding=True)

    cols = [c for c in COLUMNS_TO_REMOVE if c in raw_dataset[split].column_names]
    tokenized = raw_dataset.map(tokenize_fn, batched=True, remove_columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir="/tmp/ensemble_eval",
        per_device_eval_batch_size=64,
        report_to="none",
    )
    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, processing_class=tokenizer)
    raw_preds = trainer.predict(tokenized[split]).predictions.squeeze()
    return np.nan_to_num(raw_preds, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    """Load models, compute ensemble predictions, and print metrics."""
    parser = create_parser()
    args = parser.parse_args()

    # Load raw (non-tokenized) dataset — each model tokenizes with its own tokenizer
    if args.data_dir is not None and args.fold is not None:
        fold_dir = os.path.join(args.data_dir, "folds", f"fold_{args.fold}", "meaning")
        csmd_dataset = load_from_disk(fold_dir)
    else:
        csmd_dataset = load_dataset("davebulaval/CSMD", "meaning")

    all_preds: list[np.ndarray] = []
    for model_dir in args.model_dirs:
        print(f"Getting predictions from {model_dir}...")
        preds = get_predictions(model_dir, csmd_dataset, args.split)
        all_preds.append(preds)

    # Ensemble: simple average
    ensemble_preds = np.mean(all_preds, axis=0)
    labels = np.array(csmd_dataset[args.split]["label"])

    r2 = r2_score(labels, ensemble_preds)
    pearson_r, pearson_p = pearsonr(labels, ensemble_preds)

    print(f"\n=== Ensemble results ({len(args.model_dirs)} models) ===")
    print(f"  R2:      {r2:.4f}")
    print(f"  Pearson: {pearson_r:.4f} (p={pearson_p:.2e})")

    print("\n  Per-model R2:")
    for model_dir, preds in zip(args.model_dirs, all_preds):
        model_r2 = r2_score(labels, preds)
        print(f"    {os.path.basename(model_dir)}: {model_r2:.4f}")


if __name__ == "__main__":
    main()
