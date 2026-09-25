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
from typing import Any, Final, Optional

import click
import numpy as np

from data.schema import POLARITY_CLASSES

#: Class index to name, for reports. Inverted once here rather than in five places.
CLASS_NAMES: Final[tuple[str, ...]] = tuple(
    name for name, _ in sorted(POLARITY_CLASSES.items(), key=lambda item: item[1])
)

#: Use the development split whole. It is built stratified by class and by corpus, and at
#: about twelve thousand rows it is cheap enough to score after every epoch. Sampling it
#: here would undo that stratification at random, which is worse than the cost it saves;
#: the option survives only for a corpus built without ``--eval-per-class``.
DEFAULT_DEV_SAMPLE: Final[int] = 0


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


def head_reuse_plan(id2label: Optional[dict]) -> tuple[bool, Optional[list[int]]]:
    """Decide whether a checkpoint's existing classification head can be reused.

    Starting from a head already trained on natural language inference is close to free and
    directly on task, and the dissociation baseline says that is where polarity comes from.
    But the head is only reusable if its class ORDER is ours, and the order is not a
    convention anyone agreed on:

    * ``MoritzLaurer/DeBERTa-v3-*-mnli-*`` publishes ``{0: entailment, 1: neutral,
      2: contradiction}``, which is exactly ``POLARITY_CLASSES``;
    * ``roberta-large-mnli`` publishes ``{0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}``,
      the reverse.

    Both have three labels, so ``ignore_mismatched_sizes`` sees no mismatch and keeps the
    head either way. On the reversed one that silently trains from a permuted head, which
    is the same failure mode as the swapped SICK encoding: it does not crash, it just
    makes every number slightly wrong for a reason nobody can see.

    Args:
        id2label: The checkpoint's mapping, or None.

    Returns:
        Whether the head is on our label space, and the permutation that puts the
        checkpoint's classes into our order, or None when no reuse is possible.

    Raises:
        ValueError: If the checkpoint carries three labels that are recognisably an
            inference label space but cannot be mapped onto ours. Guessing here is how a
            permuted head gets shipped.
    """
    if not id2label or len(id2label) != len(CLASS_NAMES):
        return False, None

    names = {index: str(label).strip().lower() for index, label in id2label.items()}
    if not set(names.values()) <= set(CLASS_NAMES):
        # Not an inference label space at all, e.g. LABEL_0 / LABEL_1. Nothing to reuse and
        # nothing to warn about.
        return False, None

    if len(set(names.values())) != len(CLASS_NAMES):
        raise ValueError(f"checkpoint label space {id2label} is not a permutation of {list(CLASS_NAMES)}")

    by_name = {label: index for index, label in names.items()}
    return True, [by_name[name] for name in CLASS_NAMES]


def output_layer(model: Any) -> Any:
    """Return the final linear layer of a sequence-classification head.

    The head is not the same object across families, and the difference is invisible until
    it raises. ``DebertaV2ForSequenceClassification.classifier`` IS the linear layer, so it
    carries ``weight`` directly. ``RobertaForSequenceClassification.classifier`` is a
    ``RobertaClassificationHead``, a dense layer plus a projection, and the logits come out
    of ``out_proj``. Reaching for ``classifier.weight`` on a RoBERTa model raises, and
    ``roberta-large-mnli`` is precisely the checkpoint that needs its rows permuted.

    Args:
        model: A loaded sequence-classification model.

    Returns:
        The module whose ``weight`` rows are the classes.

    Raises:
        AttributeError: If no linear layer can be found, rather than permuting something
            that is not the head.
    """
    head = model.classifier
    for candidate in (head, getattr(head, "out_proj", None)):
        if candidate is not None and hasattr(candidate, "weight") and candidate.weight.dim() == 2:
            return candidate
    raise AttributeError(f"no output linear layer found on {type(head).__name__}")


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

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint)
    reusable, permutation = head_reuse_plan(getattr(config, "id2label", None))

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(CLASS_NAMES),
        # A checkpoint fine-tuned for a different number of labels carries a head whose
        # shape will not match. Replacing it is the intent, not an accident.
        ignore_mismatched_sizes=True,
    )

    if reusable and permutation != list(range(len(CLASS_NAMES))):
        # The head survived because the shapes agreed, but its rows are in the checkpoint's
        # order and not ours. Permuting them costs two tensor copies and saves the run from
        # starting off a head that is right about everything except which class is which.
        import torch

        layer = output_layer(model)
        with torch.no_grad():
            index = torch.tensor(permutation, device=layer.weight.device)
            layer.weight.copy_(layer.weight[index])
            if layer.bias is not None:
                layer.bias.copy_(layer.bias[index])
    model.config.id2label = {index: name for index, name in enumerate(CLASS_NAMES)}
    model.config.label2id = {name: index for index, name in enumerate(CLASS_NAMES)}

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
        "pretrained_head_reused": reusable,
        "pretrained_head_permutation": permutation,
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
