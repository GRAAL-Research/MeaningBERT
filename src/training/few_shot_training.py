"""Fine-tune a pretrained model for meaning preservation regression on CSMD."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from typing import Optional

import wandb
from datasets import DatasetDict, load_dataset, load_from_disk
from poutyne import set_seeds
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from metrics.metrics import compute_metrics, eval_compute_metrics_identical, eval_compute_metrics_unrelated

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)

NUM_EPOCH = 500

# Default learning rates per model family.
# Decoder-based and DeBERTa models tend to need lower LRs than BERT.
MODEL_FAMILY_LR: dict[str, float] = {
    "deberta": 2e-5,
    "electra": 3e-5,
    "modernbert": 5e-5,
    "gpt2": 2e-5,
    "smollm": 2e-5,
    "qwen": 1e-5,
    "gemma": 1e-5,
    "phi": 1e-5,
}
DEFAULT_LR: float = 5e-5

AUGMENTATION_DIR_MAP: dict[str, str] = {
    "none": "meaning",
    "swap": "meaning_with_swap",
    "back_translation": "meaning_with_back_translation",
}
AUGMENTATION_HF_MAP: dict[str, str] = {
    "none": "meaning",
    "swap": "meaning_with_data_augmentation",
}

# Columns added during data augmentation that are not model inputs.
COLUMNS_TO_REMOVE: list[str] = ["source"]


def get_default_lr(checkpoint: str) -> float:
    """Return a sensible default learning rate based on the checkpoint name."""
    checkpoint_lower = checkpoint.lower()
    for family, lr in MODEL_FAMILY_LR.items():
        if family in checkpoint_lower:
            return lr
    return DEFAULT_LR


def freeze_layers(model: PreTrainedModel, num_layers_to_freeze: int) -> None:
    """Freeze the first *num_layers_to_freeze* transformer layers of *model*.

    Supports BERT, DeBERTa v2/v3, ELECTRA, GPT-2, LLaMA-family, and ModernBERT.
    """
    if num_layers_to_freeze <= 0:
        return

    # Locate the layer list depending on the model architecture.
    layer_list = None
    for attr_path in [
        "base_model.encoder.layer",  # BERT, DeBERTa, ELECTRA
        "deberta.encoder.layer",  # DeBERTa v2/v3
        "model.layers",  # LLaMA, Qwen, Gemma, Phi, SmolLM
        "transformer.h",  # GPT-2
        "encoder.layers",  # ModernBERT
    ]:
        obj = model
        found = True
        for part in attr_path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                found = False
                break
        if found and hasattr(obj, "__len__"):
            layer_list = obj
            break

    if layer_list is None:
        print("WARNING: Could not find layer list for freezing. Skipping layer freeze.")
        return

    n = min(num_layers_to_freeze, len(layer_list))
    for layer in layer_list[:n]:
        for param in layer.parameters():
            param.requires_grad = False
    print(f"Froze {n}/{len(layer_list)} layers.")


def create_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Fine-tune a model for meaning preservation regression.")

    parser.add_argument("--seed", type=int, default=45, help="Random seed for training.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory containing pre-generated datasets (from prepare_datasets.py).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold index for k-fold cross-validation (0-9). Requires --data_dir.",
    )
    parser.add_argument(
        "--data_augmentation",
        type=str,
        default="swap",
        choices=["none", "swap", "back_translation"],
        help="Data augmentation variant.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model checkpoint for fine-tuning.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate. If not set, uses a per-model-family default.",
    )
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of bottom layers to freeze.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=64, help="Training batch size per device.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * accumulation).",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bfloat16 mixed precision (recommended for RTX Ada GPUs). Use --no-bf16 to disable.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use float16 mixed precision. For models where bf16 causes NaN and fp32 causes dtype errors.",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=50,
        help="Early stopping patience in epochs. 0 to disable.",
    )
    return parser


def main() -> None:
    """Entry point: parse args, load data, train, evaluate, and log artifacts."""
    parser = create_parser()
    args = parser.parse_args()

    seed: int = args.seed
    data_dir: Optional[str] = args.data_dir
    fold: Optional[int] = args.fold
    data_augmentation: str = args.data_augmentation
    checkpoint: str = args.checkpoint
    lr: float = args.learning_rate if args.learning_rate is not None else get_default_lr(checkpoint)
    num_freeze: int = args.freeze_layers
    batch_size: int = args.per_device_train_batch_size
    grad_accum: int = args.gradient_accumulation_steps
    use_bf16: bool = args.bf16
    use_fp16: bool = args.fp16
    num_workers: int = args.dataloader_num_workers
    es_patience: int = args.early_stopping_patience

    set_seeds(seed=seed)

    # --- Load datasets ---
    holdout_identical_dataset: Optional[DatasetDict] = None
    holdout_unrelated_dataset: Optional[DatasetDict] = None

    if data_dir is not None:
        base_path = os.path.join(data_dir, "folds", f"fold_{fold}") if fold is not None else data_dir
        dataset_path = os.path.join(base_path, AUGMENTATION_DIR_MAP[data_augmentation])
        print(f"Loading dataset from disk: {dataset_path}")
        csmd_dataset = load_from_disk(dataset_path)

        if fold is not None:
            # With k-fold: identical/unrelated are in the fold splits (stratified).
            # Extract them from the test set by source tag for holdout evaluation.
            test_set = csmd_dataset["test"]
            if "source" in test_set.column_names:
                holdout_identical_dataset = DatasetDict({"test": test_set.filter(lambda x: x["source"] == "identical")})
                holdout_unrelated_dataset = DatasetDict({"test": test_set.filter(lambda x: x["source"] == "unrelated")})
        else:
            holdout_identical_dataset = load_from_disk(os.path.join(data_dir, "meaning_holdout_identical"))
            holdout_unrelated_dataset = load_from_disk(os.path.join(data_dir, "meaning_holdout_unrelated"))
    else:
        if data_augmentation == "back_translation":
            raise ValueError("--data_dir is required for back_translation. Run prepare_datasets.py first.")
        hf_config = AUGMENTATION_HF_MAP[data_augmentation]
        print(f"Loading dataset from HuggingFace Hub: davebulaval/CSMD ({hf_config})")
        csmd_dataset = load_dataset("davebulaval/CSMD", hf_config)
        holdout_identical_dataset = load_dataset("davebulaval/CSMD", "meaning_holdout_identical")
        holdout_unrelated_dataset = load_dataset("davebulaval/CSMD", "meaning_holdout_unrelated")

    # --- Tokenization ---
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Decoder-based models (GPT-2, LLaMA, etc.) don't have a pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example: dict) -> dict:
        return tokenizer(example["original"], example["simplification"], truncation=True, padding=True)

    # Remove non-tensor columns before tokenization to avoid Trainer collation errors.
    cols_to_remove = [c for c in COLUMNS_TO_REMOVE if c in csmd_dataset["train"].column_names]
    tokenized_csmd_dataset = csmd_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=cols_to_remove,
        num_proc=4,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Training config ---
    checkpoint_short_name = checkpoint.replace("/", "_")
    effective_batch = batch_size * grad_accum
    fold_str = f"_fold{fold}" if fold is not None else ""
    run_name = (
        f"{checkpoint_short_name}_seed{seed}_lr{lr}_bs{effective_batch}"
        f"_freeze{num_freeze}_aug{data_augmentation}{fold_str}"
    )

    training_args = TrainingArguments(
        output_dir=f"meaning_bert_train_{checkpoint_short_name}",
        run_name=run_name,
        report_to="wandb",
        logging_strategy="epoch",
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=NUM_EPOCH,
        save_total_limit=3,
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=seed,
        metric_for_best_model="eval_loss",
        learning_rate=lr,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
    )

    # Regression head: num_labels=1
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

    # Sync model pad_token_id with tokenizer (needed for decoder-based models).
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if num_freeze > 0:
        freeze_layers(model, num_freeze)

    callbacks = []
    if es_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=es_patience))

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_csmd_dataset["train"],
        eval_dataset=tokenized_csmd_dataset["dev"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # --- Train ---
    print("----------Training start----------")
    trainer.train()

    wandb.run.config.update(
        {
            "data_augmentation": data_augmentation,
            "fold": fold,
            "checkpoint": checkpoint,
            "freeze_layers": num_freeze,
            "effective_batch_size": effective_batch,
            "early_stopping_patience": es_patience,
        }
    )
    wandb.log({"Best model checkpoint path": trainer.state.best_model_checkpoint})

    # --- Evaluate ---
    print("----------Test Set Evaluation start----------")
    test_results = trainer.evaluate(eval_dataset=tokenized_csmd_dataset["test"], metric_key_prefix="test")

    identical_results: dict = {}
    unrelated_results: dict = {}

    if holdout_identical_dataset is not None and len(holdout_identical_dataset["test"]) > 0:
        cols = [c for c in COLUMNS_TO_REMOVE if c in holdout_identical_dataset["test"].column_names]
        tok_identical = holdout_identical_dataset.map(tokenize_function, batched=True, remove_columns=cols)
        trainer.compute_metrics = eval_compute_metrics_identical
        identical_results = trainer.evaluate(
            eval_dataset=tok_identical["test"],
            metric_key_prefix="test/identical_sentences",
        )

    if holdout_unrelated_dataset is not None and len(holdout_unrelated_dataset["test"]) > 0:
        cols = [c for c in COLUMNS_TO_REMOVE if c in holdout_unrelated_dataset["test"].column_names]
        tok_unrelated = holdout_unrelated_dataset.map(tokenize_function, batched=True, remove_columns=cols)
        trainer.compute_metrics = eval_compute_metrics_unrelated
        unrelated_results = trainer.evaluate(
            eval_dataset=tok_unrelated["test"],
            metric_key_prefix="test/unrelated_sentences",
        )

    # --- Save & log artifact ---
    best_model_dir = f"meaningbert_best_model_{checkpoint_short_name}_seed{seed}{fold_str}"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    artifact_name = f"meaningbert-{checkpoint_short_name}-seed{seed}{fold_str}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=f"Best MeaningBERT model fine-tuned from {checkpoint}",
        metadata={
            "checkpoint": checkpoint,
            "seed": seed,
            "fold": fold,
            "learning_rate": lr,
            "freeze_layers": num_freeze,
            "effective_batch_size": effective_batch,
            "early_stopping_patience": es_patience,
            "data_augmentation": data_augmentation,
            "best_checkpoint_path": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric,
            "test_results": test_results,
            "holdout_identical_results": identical_results,
            "holdout_unrelated_results": unrelated_results,
        },
    )
    artifact.add_dir(best_model_dir)
    wandb.log_artifact(artifact)
    print(f"Model artifact logged to wandb: {artifact_name}")

    # Clean up intermediate checkpoints to save disk space
    output_dir = f"meaning_bert_train_{checkpoint_short_name}"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleaned up intermediate checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
