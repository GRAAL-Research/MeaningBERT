"""Open the polarity head's headline number: per corpus, and as a ranking.

The paper reports one macro-F1 over a test split that stratification made a third SICK
and two thirds VitaminC, and it reports an AUC only for the models it did not train. Two
questions follow, and this module answers both from a single forward pass.

**Where does the headline come from?** VitaminC pairs are claim-evidence pairs from
Wikipedia revisions; SICK pairs are the caption-like sentence pairs a simplification
metric actually sees. A number averaged over the two says nothing about either until it
is split.

**Does fine-tuning improve the ranking, or only the calibration?** The composition
consumes a single scalar, ``p_contra``, so what matters to it is the ordering of that
scalar and not the three-way decision. An off-the-shelf inference model already ranks
well; the comparison is only honest once the same quantity is measured on our head.

Run::

    PYTHONPATH=src python src/diagnostics/head_breakdown.py \
        --checkpoint results/polarity/nli-deberta-v3-large-none-poids/seed42/model \
        --label raw --corpus datastore/polarity/corpus --json-out results/breakdown.json
"""

from __future__ import annotations

import collections
import json
from typing import Any, Optional

import click
import numpy as np

try:  # PYTHONPATH=src.
    from data.schema import POLARITY_CLASSES
    from diagnostics.cross_task import CLASS_NAMES, Model, auc, macro_f1
except ImportError:  # pragma: no cover - script run from inside ``src/diagnostics``.
    from cross_task import CLASS_NAMES, Model, auc, macro_f1  # type: ignore
    from src.data.schema import POLARITY_CLASSES  # type: ignore

ENTAILMENT = POLARITY_CLASSES["entailment"]
CONTRADICTION = POLARITY_CLASSES["contradiction"]


def softmax(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def ranking_score(logits: np.ndarray) -> np.ndarray:
    """The scalar the composition consumes, oriented so entailment is high.

    ``1 - p_contra`` and not the entailment probability: the composition multiplies the
    magnitude by exactly this quantity, so an AUC computed on anything else would measure
    a number the scale never sees.
    """
    return 1.0 - softmax(logits)[:, CONTRADICTION]


def slice_metrics(logits: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Macro-F1, accuracy and the entailment-versus-contradiction AUC for one slice."""
    if len(truth) == 0:
        return {"n": 0, "macro_f1": float("nan"), "accuracy": float("nan"), "auc": float("nan")}
    predicted = logits.argmax(axis=1)
    score = ranking_score(logits)
    return {
        "n": int(len(truth)),
        "macro_f1": macro_f1(predicted, truth),
        "accuracy": float((predicted == truth).mean()),
        "auc": auc(score[truth == ENTAILMENT], score[truth == CONTRADICTION]),
        "per_class": {
            name: {
                "n": int((truth == index).sum()),
                "recall": (
                    float((predicted[truth == index] == index).mean()) if (truth == index).any() else float("nan")
                ),
            }
            for name, index in POLARITY_CLASSES.items()
        },
    }


def breakdown(model: Model, split) -> dict[str, Any]:
    """Score every pair once, then cut the result by corpus."""
    logits = model.logits(list(split["original"]), list(split["simplification"]))
    truth = np.array(split["polarity"], dtype=float).astype(int)
    corpora = np.array(split["corpus"])

    out: dict[str, Any] = {"overall": slice_metrics(logits, truth), "per_corpus": {}}
    for name in sorted(set(corpora.tolist())):
        keep = corpora == name
        out["per_corpus"][name] = slice_metrics(logits[keep], truth[keep])
    out["corpus_shares"] = {
        name: round(float(count) / len(corpora), 4) for name, count in collections.Counter(corpora.tolist()).items()
    }
    return out


@click.command()
@click.option("--checkpoint", required=True, help="A three-class polarity head.")
@click.option("--label", default="", help="Name used in the printout and the JSON.")
@click.option("--corpus", default="datastore/polarity/corpus", show_default=True)
@click.option("--split", default="test", show_default=True)
@click.option("--json-out", default=None)
def main(checkpoint: str, label: str, corpus: str, split: str, json_out: Optional[str]) -> None:
    """Report the head's performance overall and corpus by corpus."""
    from datasets import load_from_disk

    model = Model(checkpoint, None)
    if not model.is_polarity:
        raise click.ClickException(f"{checkpoint} is not a {len(CLASS_NAMES)}-class polarity head.")

    rows = load_from_disk(corpus)[split]
    got = breakdown(model, rows)
    got.update({"checkpoint": checkpoint, "label": label, "split": split})

    click.echo(f"{label or checkpoint}  ({split}, {got['overall']['n']} paires)")
    click.echo(f"{'tranche':<14} {'n':>7} {'macro-F1':>9} {'exactitude':>11} {'AUC':>8}")
    for name, record in [("ensemble", got["overall"])] + sorted(got["per_corpus"].items()):
        click.echo(
            f"{name:<14} {record['n']:>7} {100 * record['macro_f1']:>8.2f}%"
            f" {100 * record['accuracy']:>10.2f}% {100 * record['auc']:>7.2f}%"
        )

    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(got, handle, indent=2)
        click.echo(f"\nbrut : {json_out}")


if __name__ == "__main__":
    main()
