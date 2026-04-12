"""Ensemble evaluation: average predictions from multiple trained models."""
import argparse
import glob
import os

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def create_parser():
    parser = argparse.ArgumentParser(description="Ensemble evaluation over multiple saved models.")
    parser.add_argument(
        "--model_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to saved model directories (from trainer.save_model).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "dev"],
        help="Dataset split to evaluate on.",
    )
    return parser


def get_predictions(model_dir: str, dataset, data_collator) -> np.ndarray:
    """Load a model and return its predictions on the dataset."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir="/tmp/ensemble_eval",
        per_device_eval_batch_size=64,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    predictions = trainer.predict(dataset)
    return predictions.predictions.squeeze()


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Load dataset
    csmd_dataset = load_dataset("davebulaval/CSMD", "meaning")
    holdout_identical = load_dataset("davebulaval/CSMD", "meaning_holdout_identical")
    holdout_unrelated = load_dataset("davebulaval/CSMD", "meaning_holdout_unrelated")

    # Use tokenizer from first model for tokenization
    first_tokenizer = AutoTokenizer.from_pretrained(args.model_dirs[0])
    if first_tokenizer.pad_token is None:
        first_tokenizer.pad_token = first_tokenizer.eos_token

    def tokenize_function(example):
        return first_tokenizer(example["original"], example["simplification"], truncation=True, padding=True)

    eval_dataset = csmd_dataset.map(tokenize_function, batched=True)
    eval_identical = holdout_identical.map(tokenize_function, batched=True)
    eval_unrelated = holdout_unrelated.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(first_tokenizer)

    datasets_to_eval = {
        "test": eval_dataset[args.split],
        "holdout_identical": eval_identical["test"],
        "holdout_unrelated": eval_unrelated["test"],
    }

    for dataset_name, dataset in datasets_to_eval.items():
        all_preds = []
        for model_dir in args.model_dirs:
            print(f"Getting predictions from {model_dir} on {dataset_name}...")
            preds = get_predictions(model_dir, dataset, data_collator)
            all_preds.append(preds)

        # Ensemble: simple average
        ensemble_preds = np.mean(all_preds, axis=0)
        labels = np.array(dataset["label"])

        r2 = r2_score(labels, ensemble_preds)
        pearson_r, pearson_p = pearsonr(labels, ensemble_preds)

        print(f"\n=== Ensemble results on {dataset_name} ({len(args.model_dirs)} models) ===")
        print(f"  R2:      {r2:.4f}")
        print(f"  Pearson: {pearson_r:.4f} (p={pearson_p:.2e})")

        # Per-model results for comparison
        print(f"\n  Per-model R2:")
        for model_dir, preds in zip(args.model_dirs, all_preds):
            model_r2 = r2_score(labels, preds)
            print(f"    {os.path.basename(model_dir)}: {model_r2:.4f}")
        print()


if __name__ == "__main__":
    main()
