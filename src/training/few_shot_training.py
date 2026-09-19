"""Fine-tune a pretrained model for meaning preservation regression on CSMD."""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
from typing import Any, Optional

import torch
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

from calibration import DEFAULT_OUTPUT_HEAD, OUTPUT_HEADS, percent_from_logits, targets_for_head, unit_from_logits
from callbacks import DEFAULT_COLLAPSE_PATIENCE, PredictionCollapseCallback
from metrics.metrics import compute_metrics, eval_compute_metrics_identical, eval_compute_metrics_unrelated

try:  # PYTHONPATH=src, the documented way to run this script.
    from diagnostics.calibration_audit import COLLAPSE_STD_THRESHOLD
except ImportError:  # pragma: no cover - repository root on the path instead of ``src``.
    from src.diagnostics.calibration_audit import COLLAPSE_STD_THRESHOLD  # type: ignore[no-redef]


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf float values with None so wandb artifact metadata stays JSON-compliant."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


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
# Every non-tensor column the Trainer would choke on. The v1 fold layout only carried
# "source"; the v2 contract (src/data/CONTRACT.md) adds the provenance and scale columns,
# and they must all be dropped after tokenisation. Only input_ids, attention_mask and
# label survive.
COLUMNS_TO_REMOVE: list[str] = [
    "source",
    "item_id",
    "original",
    "simplification",
    "label_raw",
    "scale",
    "n_annotators",
    "label_std",
    "corpus",
    "domain",
    "system",
    "split_hint",
    "license",
]


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


class BoundedOutputTrainer(Trainer):
    """Trainer for the bounded output heads of correction C3.

    The v1 head is a single unbounded linear output trained with MSE against 0-100
    targets: nothing tells it the scale is bounded, and the sweep answered by compressing
    its predictions into a quarter of the label amplitude. A bounded head removes the
    cause. Two things have to happen for that, and both happen here:

    * the loss is computed on the unit scale the head lives on, through
      :func:`calibration.unit_from_logits` and :func:`calibration.targets_for_head`;
    * the predictions handed to ``compute_metrics`` are put back on the 0-100 scale by
      :func:`calibration.percent_from_logits`, so every reported metric stays comparable
      to the published article.

    The dataset is never touched: the labels stay on the 0-100 scale end to end.

    Args:
        *args: Forwarded to ``Trainer``.
        output_head: One of ``calibration.OUTPUT_HEADS``, minus ``linear``, which uses a
            plain ``Trainer`` instead.
        **kwargs: Forwarded to ``Trainer``.
    """

    def __init__(self, *args: Any, output_head: str = DEFAULT_OUTPUT_HEAD, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.output_head = output_head

    def compute_loss(  # pylint: disable=arguments-differ,unused-argument
        self, model, inputs, return_outputs=False, **kwargs
    ):
        """Compute the MSE on the unit scale of the configured head.

        Args:
            model: The model being trained.
            inputs: The batch, labels included, on the 0-100 scale.
            return_outputs: Whether to return the model outputs alongside the loss.
            **kwargs: Extras passed by newer ``Trainer`` versions, such as
                ``num_items_in_batch``. Ignored on purpose: the loss is a plain mean.

        Returns:
            The loss, or ``(loss, outputs)``.
        """
        labels = inputs.get("labels")
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits.squeeze(-1)
        predictions = unit_from_logits(logits, self.output_head)
        targets = targets_for_head(labels.to(predictions.dtype), self.output_head)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        return (loss, outputs) if return_outputs else loss


def make_logits_to_percent(output_head: str):
    """Build the ``preprocess_logits_for_metrics`` hook putting predictions back on 0-100.

    Args:
        output_head: One of ``calibration.OUTPUT_HEADS``.

    Returns:
        A callable ``(logits, labels) -> scores``, or ``None`` for the linear head, which
        already reports on the 0-100 scale.
    """
    if output_head == "linear":
        return None

    def _to_percent(logits, labels):  # pylint: disable=unused-argument
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return percent_from_logits(logits, output_head)

    return _to_percent


def assert_bf16_is_supported() -> None:
    """Fail loudly when bf16 is requested on a GPU that cannot do it.

    bf16 needs compute capability 8.0. The v2 training server, renard, runs a Quadro P5000
    (Pascal, capability 6.1), where the flag either errors out deep in accelerate or falls
    back to fp32 without saying so. Neither is acceptable for a run whose numbers end up in
    a table.

    Raises:
        SystemExit: If no CUDA device is visible, or its capability is below 8.0.
    """
    if not torch.cuda.is_available():
        raise SystemExit("--bf16 was requested but no CUDA device is visible.")
    major, minor = torch.cuda.get_device_capability(0)
    if major < 8:
        raise SystemExit(
            f"--bf16 was requested but {torch.cuda.get_device_name(0)} has compute capability {major}.{minor}, "
            f"below the 8.0 bf16 needs. Drop the flag to train in fp32, which is the default."
        )


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
        "--variant_path",
        type=str,
        default=None,
        help=(
            "Path to one variant produced by src/data/build_corpus.py, holding train/dev/test/sanity "
            "already split by source sentence and already augmented. Takes precedence over --data_dir. "
            "This is the v2 path; --data_dir stays for the v1 fold layout."
        ),
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
        default=False,
        help=(
            "Use bfloat16 mixed precision. Needs compute capability 8.0 (Ampere or newer) and is refused "
            "otherwise. The default is fp32: the v2 training server runs a Pascal card, which has no bf16."
        ),
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
    parser.add_argument(
        "--output_head",
        type=str,
        default=DEFAULT_OUTPUT_HEAD,
        choices=list(OUTPUT_HEADS),
        help=(
            "Output layer (correction C3). 'linear' is the unbounded v1 head and the default, so an existing "
            "command is unchanged. 'sigmoid' trains 100*sigmoid(logit), bounded at every step. 'normalized' "
            "trains a linear head against targets divided by 100 and rescales at evaluation. The reported "
            "metrics stay on the 0-100 scale in all three cases."
        ),
    )
    parser.add_argument(
        "--collapse_patience",
        type=int,
        default=DEFAULT_COLLAPSE_PATIENCE,
        help=(
            "Number of consecutive evaluations with a degenerate prediction spread before the run is stopped "
            "(correction C2). 0 restores the pre-C2 behaviour, where a collapsed run trains to the end."
        ),
    )
    parser.add_argument(
        "--collapse_std_threshold",
        type=float,
        default=COLLAPSE_STD_THRESHOLD,
        help=(
            "Prediction standard deviation below which an evaluation counts as degenerate. The default is the "
            "threshold the H1 audit uses; a healthy run of the sweep sits between 9 and 17."
        ),
    )
    return parser


def main() -> None:
    """Entry point: parse args, load data, train, evaluate, and log artifacts."""
    parser = create_parser()
    args = parser.parse_args()

    seed: int = args.seed
    data_dir: Optional[str] = args.data_dir
    variant_path: Optional[str] = args.variant_path
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
    collapse_patience: int = args.collapse_patience
    collapse_std_threshold: float = args.collapse_std_threshold
    output_head: str = args.output_head

    if use_bf16:
        assert_bf16_is_supported()

    set_seeds(seed=seed)

    # --- Load datasets ---
    holdout_identical_dataset: Optional[DatasetDict] = None
    holdout_unrelated_dataset: Optional[DatasetDict] = None

    if variant_path is not None:
        # v2 layout: one directory holding train/dev/test/sanity, already split by source
        # sentence and already augmented. The sanity split is a genuine holdout here, which
        # is what docs/H5-fuite-par-phrase-source.md shows v1 never had.
        print(f"Loading v2 variant from disk: {variant_path}")
        loaded = load_from_disk(variant_path)
        csmd_dataset = DatasetDict({split: loaded[split] for split in ("train", "dev", "test")})
        sanity = loaded["sanity"]
        holdout_identical_dataset = DatasetDict({"test": sanity.filter(lambda r: r["source"] == "identical")})
        holdout_unrelated_dataset = DatasetDict({"test": sanity.filter(lambda r: r["source"] == "unrelated")})
        print(
            f"  train={len(csmd_dataset['train'])} dev={len(csmd_dataset['dev'])} "
            f"test={len(csmd_dataset['test'])} identical={len(holdout_identical_dataset['test'])} "
            f"unrelated={len(holdout_unrelated_dataset['test'])}"
        )
    elif data_dir is not None:
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
    head_str = "" if output_head == DEFAULT_OUTPUT_HEAD else f"_head{output_head}"
    run_name = (
        f"{checkpoint_short_name}_seed{seed}_lr{lr}_bs{effective_batch}"
        f"_freeze{num_freeze}_aug{data_augmentation}{fold_str}{head_str}"
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

    # Regression head: num_labels=1.
    # Explicit torch_dtype prevents models that default to float16 in their config (e.g. phi-2)
    # from being loaded in FP16 while bf16 training is requested, which causes accelerate to
    # create a GradScaler and then fail with "Attempting to unscale FP16 gradients."
    if use_bf16:
        model_dtype = torch.bfloat16
    elif use_fp16:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1, torch_dtype=model_dtype)

    # Sync model pad_token_id with tokenizer (needed for decoder-based models).
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if num_freeze > 0:
        freeze_layers(model, num_freeze)

    callbacks = []
    if es_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=es_patience))

    # C2: stop a run whose predictions collapsed to a constant, instead of letting it burn
    # the GPU to the last epoch and report a plausible-looking RMSE.
    collapse_callback: Optional[PredictionCollapseCallback] = None
    if collapse_patience > 0:
        collapse_callback = PredictionCollapseCallback(
            std_threshold=collapse_std_threshold,
            patience=collapse_patience,
        )
        callbacks.append(collapse_callback)

    # C3: the head the score is read through must travel with the checkpoint, otherwise
    # whoever loads it later reads raw logits as if they were a 0-100 score.
    model.config.meaningbert_output_head = output_head

    trainer_kwargs: dict[str, Any] = {
        "train_dataset": tokenized_csmd_dataset["train"],
        "eval_dataset": tokenized_csmd_dataset["dev"],
        "data_collator": data_collator,
        "processing_class": tokenizer,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }
    if output_head == DEFAULT_OUTPUT_HEAD:
        # Untouched v1 path: a plain Trainer, the model's own MSE on the 0-100 scale.
        trainer = Trainer(model, training_args, **trainer_kwargs)
    else:
        trainer = BoundedOutputTrainer(
            model,
            training_args,
            output_head=output_head,
            preprocess_logits_for_metrics=make_logits_to_percent(output_head),
            **trainer_kwargs,
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
            "collapse_patience": collapse_patience,
            "collapse_std_threshold": collapse_std_threshold,
            "output_head": output_head,
            "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
        }
    )

    collapse_reason = collapse_callback.stop_reason if collapse_callback is not None else None
    if collapse_reason is not None:
        print(f"WARNING: training stopped early because {collapse_reason}. This run is not usable.")
        wandb.log({"train/collapse_stop_reason": collapse_reason, "train/collapsed": 1.0})
    else:
        wandb.log({"train/collapsed": 0.0})
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
        metadata=_sanitize_for_json(
            {
                "checkpoint": checkpoint,
                "seed": seed,
                "fold": fold,
                "learning_rate": lr,
                "freeze_layers": num_freeze,
                "effective_batch_size": effective_batch,
                "early_stopping_patience": es_patience,
                "collapse_patience": collapse_patience,
                "collapse_stop_reason": collapse_reason,
                "output_head": output_head,
                "data_augmentation": data_augmentation,
                "best_checkpoint_path": trainer.state.best_model_checkpoint,
                "best_eval_loss": trainer.state.best_metric,
                "test_results": test_results,
                "holdout_identical_results": identical_results,
                "holdout_unrelated_results": unrelated_results,
            }
        ),
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
