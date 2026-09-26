"""Score one checkpoint on BOTH campaigns' tasks, so the two can be read side by side.

The v2 and v3 campaigns ask different questions and, until now, answered them with
different numbers on different data. That makes the interesting comparison impossible: a
v2 model is reported in Pearson on meaning preservation, a v3 model in macro-F1 on
polarity, and nobody can say what either costs the other.

This puts every checkpoint through both:

**The v2 task, meaning preservation.** Pearson against human judgements on the v2 test
split. A regression checkpoint answers directly. A three-class polarity checkpoint has no
such output, so its score is ``100 x P(entailment)``: the probability the candidate follows
from the source, which is the closest thing a polarity head has to "the meaning survived".
It is a derived quantity and is labelled as one, never presented as what the model was
trained to produce.

**The v3 task, polarity.** On the v3 test split. Two numbers, chosen so that models with
different output shapes stay comparable:

* **AUC**, entailment against contradiction, which any scalar can produce. It is the only
  axis on which a v2 regression model and a v3 classifier can be compared at all.
* **macro-F1**, which needs three classes and is therefore reported for polarity heads
  only, blank for the others rather than faked.

Run::

    PYTHONPATH=src python src/diagnostics/cross_task.py \\
        --checkpoint davebulaval/MeaningBERT --label "v1 publie" \\
        --checkpoint davebulaval/MeaningBERT --subfolder large --label "v2 large"
"""

from __future__ import annotations

import json
from typing import Any, Optional

import click
import numpy as np
from scipy.stats import pearsonr, rankdata

try:  # PYTHONPATH=src.
    from data.schema import POLARITY_CLASSES
except ImportError:  # pragma: no cover
    from src.data.schema import POLARITY_CLASSES  # type: ignore

#: Class index to name, in schema order.
CLASS_NAMES = tuple(name for name, _ in sorted(POLARITY_CLASSES.items(), key=lambda item: item[1]))


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Share of (positive, negative) couples the score orders correctly, ties at half.

    Mid-ranks, not ordinal ranks: a clamped head saturates at exactly 0 and 100, so ties
    between the two classes are the common case rather than an edge case.
    """
    n_p, n_n = len(positive), len(negative)
    if not n_p or not n_n:
        return float("nan")
    ranks = rankdata(np.concatenate([positive, negative]))
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def macro_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Unweighted mean F1 over the three classes."""
    scores = []
    for index in range(len(CLASS_NAMES)):
        true_positive = int(((predictions == index) & (labels == index)).sum())
        predicted = int((predictions == index).sum())
        actual = int((labels == index).sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


class Model:
    """A checkpoint, asked for whatever it can answer.

    Two shapes exist in this project and they answer different questions. A regression
    checkpoint emits one number per pair, already on the 0-100 meaning scale. A polarity
    checkpoint emits three logits. Rather than pretend they are the same, this exposes what
    each can produce and lets the caller leave the rest blank.
    """

    def __init__(self, checkpoint: str, subfolder: Optional[str], batch_size: int = 32) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        where = {"subfolder": subfolder} if subfolder else {}
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **where)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, **where).to(self.device).eval()
        self.batch_size = batch_size

        config = AutoConfig.from_pretrained(checkpoint, **where)
        self.n_labels = int(getattr(config, "num_labels", 1) or 1)
        self.is_polarity = self.n_labels == len(CLASS_NAMES)
        # Regression checkpoints say which output head they were trained with; v1 predates
        # the field and its logit is already the score.
        self.head = str(getattr(config, "meaningbert_output_head", "linear") or "linear")
        self.max_length = min(int(getattr(config, "max_position_embeddings", 512) or 512), 512)

    def _logits(self, left: list[str], right: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(left), self.batch_size):
            batch = self.tokenizer(
                left[start : start + self.batch_size],
                right[start : start + self.batch_size],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.no_grad():
                out.append(self.model(**batch).logits.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.n_labels))

    def meaning_score(self, left: list[str], right: list[str]) -> np.ndarray:
        """A 0-100 meaning-preservation score, whatever the checkpoint's shape."""
        logits = self._logits(left, right)
        if self.is_polarity:
            # Derived, and labelled as derived: the probability the candidate follows from
            # the source is the closest a polarity head comes to "the meaning survived".
            exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
            return 100.0 * probabilities[:, POLARITY_CLASSES["entailment"]]

        values = logits.squeeze(-1)
        if self.head == "sigmoid":
            values = 100.0 / (1.0 + np.exp(-values))
        elif self.head in ("normalized", "clamped"):
            values = values * 100.0
        return np.clip(values, 0.0, 100.0)

    def agreement_score(self, left: list[str], right: list[str]) -> np.ndarray:
        """A scalar that should be high on entailment and low on contradiction."""
        logits = self._logits(left, right)
        if self.is_polarity:
            return logits[:, POLARITY_CLASSES["entailment"]] - logits[:, POLARITY_CLASSES["contradiction"]]
        return self.meaning_score(left, right)

    def polarity_classes(self, left: list[str], right: list[str]) -> Optional[np.ndarray]:
        """Predicted class indices, or None for a checkpoint that has no classes."""
        if not self.is_polarity:
            return None
        return self._logits(left, right).argmax(axis=1)


def evaluate(model: Model, v2_test, v3_test) -> dict[str, Any]:
    """Score *model* on both tasks."""
    predicted = model.meaning_score(list(v2_test["original"]), list(v2_test["simplification"]))
    labels = np.array(v2_test["label"], dtype=float)
    keep = np.isfinite(predicted) & np.isfinite(labels)
    pearson = float(pearsonr(predicted[keep], labels[keep])[0]) if keep.sum() > 2 else float("nan")

    left, right = list(v3_test["original"]), list(v3_test["simplification"])
    truth = np.array(v3_test["polarity"], dtype=float).astype(int)
    agreement = model.agreement_score(left, right)
    entail = agreement[truth == POLARITY_CLASSES["entailment"]]
    contra = agreement[truth == POLARITY_CLASSES["contradiction"]]

    classes = model.polarity_classes(left, right)
    return {
        "is_polarity_head": model.is_polarity,
        "v2_pearson": pearson,
        "v2_n": int(keep.sum()),
        "v3_auc_entail_vs_contra": auc(entail, contra),
        "v3_amplitude_share": float(entail.mean() - contra.mean()) / 100.0 if not model.is_polarity else None,
        "v3_macro_f1": macro_f1(classes, truth) if classes is not None else None,
        "v3_accuracy": float((classes == truth).mean()) if classes is not None else None,
        "v3_n": len(truth),
    }


@click.command()
@click.option("--checkpoint", "checkpoints", multiple=True, required=True)
@click.option("--subfolder", "subfolders", multiple=True, help="Aligned with --checkpoint; '' for none.")
@click.option("--label", "labels", multiple=True, help="Aligned with --checkpoint.")
@click.option("--v2-corpus", default="data/v2/d_full", show_default=True)
@click.option("--v3-corpus", default="datastore/polarity/corpus", show_default=True)
@click.option("--json-out", default=None)
def main(checkpoints, subfolders, labels, v2_corpus: str, v3_corpus: str, json_out: Optional[str]) -> None:
    """Score every checkpoint on both tasks and print one row each."""
    from datasets import load_from_disk

    subfolders = list(subfolders) + [""] * (len(checkpoints) - len(subfolders))
    labels = list(labels) + list(checkpoints[len(labels) :])

    v2_test = load_from_disk(v2_corpus)["test"]
    v3_test = load_from_disk(v3_corpus)["test"]
    click.echo(f"tache v2 : {len(v2_test)} paires   tache v3 : {len(v3_test)} paires\n")

    header = ("modele", "tache v2 Pearson", "v3 AUC", "v3 macro-F1", "v3 exact.")
    click.echo("%-34s %16s %8s %12s %10s" % header)

    findings = []
    for checkpoint, subfolder, label in zip(checkpoints, subfolders, labels):
        got = evaluate(Model(checkpoint, subfolder or None), v2_test, v3_test)
        got.update({"checkpoint": checkpoint, "subfolder": subfolder, "label": label})
        findings.append(got)
        click.echo(
            "%-34s %16.4f %8.4f %12s %10s"
            % (
                label[:34],
                got["v2_pearson"],
                got["v3_auc_entail_vs_contra"],
                f"{got['v3_macro_f1']:.4f}" if got["v3_macro_f1"] is not None else "--",
                f"{got['v3_accuracy']:.4f}" if got["v3_accuracy"] is not None else "--",
            )
        )

    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(findings, handle, indent=2)
        click.echo(f"\nbrut : {json_out}")


if __name__ == "__main__":
    main()
