import argparse
import logging
import os

import wandb
from datasets import DatasetDict, load_dataset, load_from_disk
from poutyne import set_seeds
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    EarlyStoppingCallback,
)

from metrics.metrics import compute_metrics, eval_compute_metrics_identical, eval_compute_metrics_unrelated

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)

num_epoch = 500

# Default learning rates per model family.
# Decoder-based and DeBERTa models tend to need lower LRs than BERT.
MODEL_FAMILY_LR = {
    "deberta": 2e-5,
    "electra": 3e-5,
    "modernbert": 5e-5,
    "gpt2": 2e-5,
    "smollm": 2e-5,
    "qwen": 1e-5,
    "gemma": 1e-5,
    "phi": 1e-5,
}
DEFAULT_LR = 5e-5


def get_default_lr(checkpoint: str) -> float:
    """Return a sensible default LR based on the checkpoint name."""
    checkpoint_lower = checkpoint.lower()
    for family, lr in MODEL_FAMILY_LR.items():
        if family in checkpoint_lower:
            return lr
    return DEFAULT_LR


def freeze_layers(model, num_layers_to_freeze: int) -> None:
    """Freeze the first `num_layers_to_freeze` transformer layers of the model."""
    if num_layers_to_freeze <= 0:
        return

    # Locate the layer list depending on the model architecture.
    layer_list = None
    for attr_path in [
        "base_model.encoder.layer",       # BERT, DeBERTa, ELECTRA
        "deberta.encoder.layer",           # DeBERTa v2/v3
        "model.layers",                    # LLaMA, Qwen, Gemma, Phi, SmolLM
        "transformer.h",                   # GPT-2
        "encoder.layers",                  # ModernBERT
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
        print(f"WARNING: Could not find layer list for freezing. Skipping layer freeze.")
        return

    n = min(num_layers_to_freeze, len(layer_list))
    for layer in layer_list[:n]:
        for param in layer.parameters():
            param.requires_grad = False
    print(f"Froze {n}/{len(layer_list)} layers.")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=45,
        help="The seed to use for training.",
    )

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory containing pre-generated datasets (from prepare_datasets.py). "
             "When set, loads from disk instead of HuggingFace Hub.",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold index for k-fold cross-validation (0-9). "
             "Requires --data_dir. Loads from data_dir/folds/fold_N/.",
    )

    parser.add_argument(
        "--data_augmentation",
        type=str,
        default="swap",
        choices=["none", "swap", "back_translation"],
        help="Data augmentation variant: 'none' (base), 'swap' (commutative), "
             "'back_translation' (swap + back-translation).",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="bert-base-uncased",
        help="The pretrained model checkpoint to use for fine-tuning.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate. If not set, uses a per-model-family default.",
    )

    parser.add_argument(
        "--freeze_layers",
        type=int,
        default=0,
        help="Number of bottom transformer layers to freeze.",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Training batch size per device.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * accumulation).",
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=50,
        help="Early stopping patience in epochs. 0 to disable.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    seed = args.seed
    root = args.root
    data_dir = args.data_dir
    fold = args.fold
    data_augmentation = args.data_augmentation
    checkpoint = args.checkpoint
    lr = args.learning_rate if args.learning_rate is not None else get_default_lr(checkpoint)
    num_freeze = args.freeze_layers
    batch_size = args.per_device_train_batch_size
    grad_accum = args.gradient_accumulation_steps
    es_patience = args.early_stopping_patience

    set_seeds(seed=seed)

    # Map augmentation variant to directory names
    AUGMENTATION_DIR_MAP = {
        "none": "meaning",
        "swap": "meaning_with_swap",
        "back_translation": "meaning_with_back_translation",
    }
    AUGMENTATION_HF_MAP = {
        "none": "meaning",
        "swap": "meaning_with_data_augmentation",
    }

    if data_dir is not None:
        # Build path: data_dir/folds/fold_N/<augmentation> or data_dir/<augmentation>
        if fold is not None:
            base_path = os.path.join(data_dir, "folds", f"fold_{fold}")
        else:
            base_path = data_dir
        dataset_path = os.path.join(base_path, AUGMENTATION_DIR_MAP[data_augmentation])
        print(f"Loading dataset from disk: {dataset_path}")
        csmd_dataset = load_from_disk(dataset_path)

        if fold is not None:
            # With k-fold: identical and unrelated are in the fold splits (stratified).
            # Extract them from the test set by source tag for holdout evaluation.
            test_set = csmd_dataset["test"]
            if "source" in test_set.column_names:
                identical_mask = [s == "identical" for s in test_set["source"]]
                unrelated_mask = [s == "unrelated" for s in test_set["source"]]
                holdout_identical_dataset = DatasetDict({
                    "test": test_set.select([i for i, m in enumerate(identical_mask) if m])
                })
                holdout_unrelated_dataset = DatasetDict({
                    "test": test_set.select([i for i, m in enumerate(unrelated_mask) if m])
                })
            else:
                holdout_identical_dataset = None
                holdout_unrelated_dataset = None
        else:
            holdout_identical_dataset = load_from_disk(os.path.join(data_dir, "meaning_holdout_identical"))
            holdout_unrelated_dataset = load_from_disk(os.path.join(data_dir, "meaning_holdout_unrelated"))
    else:
        # Fallback: download from HuggingFace Hub (no back_translation available)
        if data_augmentation == "back_translation":
            raise ValueError("--data_dir is required for back_translation. Run prepare_datasets.py first.")
        hf_config = AUGMENTATION_HF_MAP[data_augmentation]
        print(f"Loading dataset from HuggingFace Hub: davebulaval/CSMD ({hf_config})")
        csmd_dataset = load_dataset("davebulaval/CSMD", hf_config)
        holdout_identical_dataset = load_dataset("davebulaval/CSMD", "meaning_holdout_identical")
        holdout_unrelated_dataset = load_dataset("davebulaval/CSMD", "meaning_holdout_unrelated")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Decoder-based models (GPT-2, LLaMA, etc.) don't have a pad token by default.
    # Use eos_token as pad_token for proper batching and padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        return tokenizer(example["original"], example["simplification"], truncation=True, padding=True)

    tokenized_csmd_dataset = csmd_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    checkpoint_short_name = checkpoint.replace("/", "_")
    effective_batch = batch_size * grad_accum
    fold_str = f"_fold{fold}" if fold is not None else ""
    run_name = f"{checkpoint_short_name}_seed{seed}_lr{lr}_bs{effective_batch}_freeze{num_freeze}_aug{data_augmentation}{fold_str}"

    training_args = TrainingArguments(
        output_dir=f"meaning_bert_train_{checkpoint_short_name}",
        run_name=run_name,
        report_to="wandb",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epoch,
        save_total_limit=3,
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=seed,
        metric_for_best_model="eval_loss",
        learning_rate=lr,
    )

    # num_labels to 1 to create a regression head
    # REF: https://discuss.huggingface.co/t/fine-tune-bert-and-camembert-for-regression-problem/332/17
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

    # Sync model pad_token_id with tokenizer (needed for decoder-based models).
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Layer freezing
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
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    print("----------Training start----------")
    trainer.train()

    wandb.run.config.update({
        "data_augmentation": data_augmentation,
        "fold": fold,
        "checkpoint": checkpoint,
        "freeze_layers": num_freeze,
        "effective_batch_size": effective_batch,
        "early_stopping_patience": es_patience,
    })
    wandb.log({"Best model checkpoint path": trainer.state.best_model_checkpoint})

    print("----------Test Set Evaluation start----------")
    test_results = trainer.evaluate(eval_dataset=tokenized_csmd_dataset["test"], metric_key_prefix="test")

    # Evaluate holdout splits (identical / unrelated sentences)
    identical_results = {}
    unrelated_results = {}

    if holdout_identical_dataset is not None and len(holdout_identical_dataset["test"]) > 0:
        tokenize_holdout_identical_dataset = holdout_identical_dataset.map(tokenize_function, batched=True)
        trainer.compute_metrics = eval_compute_metrics_identical
        identical_results = trainer.evaluate(
            eval_dataset=tokenize_holdout_identical_dataset["test"],
            metric_key_prefix="test/identical_sentences",
        )

    if holdout_unrelated_dataset is not None and len(holdout_unrelated_dataset["test"]) > 0:
        tokenize_holdout_unrelated_dataset = holdout_unrelated_dataset.map(tokenize_function, batched=True)
        trainer.compute_metrics = eval_compute_metrics_unrelated
        unrelated_results = trainer.evaluate(
            eval_dataset=tokenize_holdout_unrelated_dataset["test"],
            metric_key_prefix="test/unrelated_sentences",
        )

    # Save best model locally
    best_model_dir = f"meaningbert_best_model_{checkpoint_short_name}_{seed}"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # Log best model as wandb artifact for later ensemble/deployment
    artifact = wandb.Artifact(
        name=f"meaningbert-{checkpoint_short_name}-seed{seed}",
        type="model",
        description=f"Best MeaningBERT model fine-tuned from {checkpoint}",
        metadata={
            "checkpoint": checkpoint,
            "seed": seed,
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
    print(f"Model artifact logged to wandb: meaningbert-{checkpoint_short_name}-seed{seed}")


if __name__ == "__main__":
    main()
