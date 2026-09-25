"""Train the v3 polarity head: three classes, on its own, before it composes anything.

Experiment 3 of ``docs/v3-echelle-signee.md``. The question it answers is narrow on
purpose: **does a polarity classifier work at all on this corpus**, measured as a
classifier, before anyone multiplies its output by a magnitude and calls the product a
signed score. A composition built on a head that cannot tell a contradiction from a
paraphrase would fail in a way no end-to-end number explains.

Why a separate script rather than a flag on ``few_shot_training.py``. That script is 761
lines built around a regression target with a bounded output head, a 0-100 rescaling and
sanity checks defined on that scale. None of it applies to a three-class classifier, and
threading a second mode through it would put the v2 pipeline at risk for no gain.

What it reports, beyond accuracy:

- **the confusion matrix**, because the errors that matter are not symmetric. Calling a
  contradiction "neutral" costs the sign; calling a neutral pair a contradiction invents
  one. Accuracy hides which is happening.
- **the two held-out probes**, MoNLI and NaN-NLI, which never appear in training. They
  are where the corpus cannot help: minimal pairs under negation. A head that scores well
  on VitaminC and at chance on NaN-NLI has learned the corpus, not the relation.

Run::

    PYTHONPATH=src python src/training/train_polarity.py \\
        --corpus datastore/polarity --checkpoint microsoft/deberta-v3-large --seed 42
"""

from __future__ import annotations

import json
import os
from typing import Any, Final

import click
import numpy as np

from data.schema import POLARITY_CLASSES

#: Class index to name, for reports. Inverted once here rather than in five places.
CLASS_NAMES: Final[tuple[str, ...]] = tuple(
    name for name, _ in sorted(POLARITY_CLASSES.items(), key=lambda item: item[1])
)

#: The dev split is VitaminC's whole validation set, 63 000 rows. Scoring it after every
#: epoch costs more than the epoch. Model selection runs on a fixed sample of it; the final
#: numbers are computed on the complete test split, which is never sampled.
DEFAULT_DEV_SAMPLE: Final[int] = 4_000


def confusion(predictions: np.ndarray, labels: np.ndarray) -> list[list[int]]:
    """Confusion matrix as ``matrix[true][predicted]``, in class-index order.

    Args:
        predictions: Predicted class indices.
        labels: True class indices.

    Returns:
        A ``k x k`` matrix of counts, with rows reading as "what the true class became".
    """
    size = len(CLASS_NAMES)
    matrix = [[0] * size for _ in range(size)]
    for true, predicted in zip(labels.astype(int), predictions.astype(int)):
        matrix[true][predicted] += 1
    return matrix


def per_class_f1(matrix: list[list[int]]) -> dict[str, float]:
    """F1 per class, from the confusion matrix.

    Macro-F1 is reported beside accuracy because the classes are not balanced in the
    evaluation splits: VitaminC's test set is 48 % entailment and 17 % neutral, so a head
    that never predicts neutral still scores respectably on accuracy alone.

    Args:
        matrix: Output of :func:`confusion`.

    Returns:
        One F1 per class name, plus ``macro``.
    """
    scores: dict[str, float] = {}
    for index, name in enumerate(CLASS_NAMES):
        true_positive = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        actual = sum(matrix[index])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        scores[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    scores["macro"] = sum(scores[name] for name in CLASS_NAMES) / len(CLASS_NAMES)
    return scores


def summarise(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class F1 and the confusion matrix, in one dict."""
    matrix = confusion(predictions, labels)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(CLASS_NAMES)))
    f1 = per_class_f1(matrix)
    return {
        "n": total,
        "accuracy": correct / total if total else float("nan"),
        "macro_f1": f1["macro"],
        "f1": {name: f1[name] for name in CLASS_NAMES},
        "confusion": matrix,
        "class_names": list(CLASS_NAMES),
    }


def compute_metrics(eval_prediction) -> dict[str, float]:
    """Trainer hook: accuracy and macro-F1, the two scalars it can track."""
    logits, labels = eval_prediction
    predictions = np.asarray(logits).argmax(axis=-1)
    got = summarise(predictions, np.asarray(labels))
    return {"accuracy": got["accuracy"], "macro_f1": got["macro_f1"]}


@click.command()
@click.option("--corpus", required=True, help="Directory produced by build_polarity_corpus.py.")
@click.option("--checkpoint", default="microsoft/deberta-v3-large", show_default=True)
@click.option("--output-dir", required=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--epochs", default=2.0, show_default=True)
@click.option("--lr", default=1e-5, show_default=True)
@click.option("--batch-size", default=16, show_default=True)
@click.option("--grad-accum", default=2, show_default=True)
@click.option("--max-length", default=256, show_default=True)
@click.option("--dev-sample", default=DEFAULT_DEV_SAMPLE, show_default=True, help="0 to use the whole dev split.")
@click.option("--keep-model/--no-keep-model", default=True, show_default=True)
def main(  # noqa: PLR0913 - a training entry point is a pile of knobs by nature
    corpus: str,
    checkpoint: str,
    output_dir: str,
    seed: int,
    epochs: float,
    lr: float,
    batch_size: int,
    grad_accum: int,
    max_length: int,
    dev_sample: int,
    keep_model: bool,
) -> None:
    """Train one polarity head and write its metrics, including the held-out probes."""
    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    splits = load_from_disk(f"{corpus}/corpus")
    probes = load_from_disk(f"{corpus}/probes")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def prepare(dataset):
        # The label column the Trainer reads must be an integer class index; ``polarity``
        # is a float because the schema uses NaN to mean "not annotated", which an int
        # column cannot express.
        encoded = dataset.map(
            lambda batch: tokenizer(
                batch["original"], batch["simplification"], truncation=True, max_length=max_length
            ),
            batched=True,
        )
        encoded = encoded.map(lambda batch: {"labels": [int(value) for value in batch["polarity"]]}, batched=True)
        return encoded.select_columns(["input_ids", "attention_mask", "labels"])

    train = prepare(splits["train"]).shuffle(seed=seed)
    dev_full = splits["dev"]
    dev = dev_full.shuffle(seed=seed).select(range(min(dev_sample, len(dev_full)))) if dev_sample else dev_full
    dev = prepare(dev)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(CLASS_NAMES),
        # A checkpoint already fine-tuned for a different number of labels carries a head
        # whose shape will not match. Replacing it is the intent, not an accident.
        ignore_mismatched_sizes=True,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"{output_dir}/hf",
            seed=seed,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=grad_accum,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=100,
            report_to=[],
            fp16=False,
            bf16=False,
        ),
        train_dataset=train,
        eval_dataset=dev,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    def evaluate(dataset) -> dict[str, Any]:
        output = trainer.predict(prepare(dataset))
        return summarise(np.asarray(output.predictions).argmax(axis=-1), np.asarray(output.label_ids))

    results: dict[str, Any] = {
        "checkpoint": checkpoint,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "train_rows": len(train),
        # The full test split, never sampled: the dev sample is a speed decision for model
        # selection and must not leak into the reported numbers.
        "test": evaluate(splits["test"]),
        "probes": {name: evaluate(probes[name]) for name in probes},
    }

    with open(f"{output_dir}/metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    if keep_model:
        trainer.save_model(f"{output_dir}/model")
        tokenizer.save_pretrained(f"{output_dir}/model")

    test = results["test"]
    click.echo(f"test     exactitude {test['accuracy']:.4f}  macro-F1 {test['macro_f1']:.4f}  {test['f1']}")
    for name, got in results["probes"].items():
        click.echo(f"sonde {name:8} exactitude {got['accuracy']:.4f}  macro-F1 {got['macro_f1']:.4f}")
    click.echo(f"metriques : {output_dir}/metrics.json")

    # The process exits right after, but the cache release makes the footprint readable
    # when several runs share a card.
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
