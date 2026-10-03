"""The signed scale applied to real simplification pairs, not to inference pairs.

Every number elsewhere in the v3 work is measured on corpora built for inference: SICK
captions and VitaminC claim-evidence pairs. The metric, however, is meant for the output
of a simplification system. This module scores the human-annotated preservation corpus
with the composed scale and reports what the sign does there: how many real
simplifications fall below zero, and whether the ones that do are the ones humans rated
lowest.

A simplification corpus has no polarity gold, so nothing here is an accuracy. What it
gives is the distribution the scale produces in the setting it was designed for, and the
correlation it keeps with the human rating it is meant to replace.

Run::

    PYTHONPATH=src python src/diagnostics/signed_on_simplification.py \\
        --magnitude davebulaval/MeaningBERT --polarity <checkpoint> \\
        --json-out results/v3/signed-simplification.json
"""

from __future__ import annotations

import json
from typing import Optional

import click
import numpy as np
from scipy.stats import pearsonr

try:  # PYTHONPATH=src.
    from data.schema import POLARITY_CLASSES
    from diagnostics.composition import compose
    from diagnostics.cross_task import Model
except ImportError:  # pragma: no cover - script run from inside ``src/diagnostics``.
    from composition import compose  # type: ignore
    from cross_task import Model  # type: ignore
    from src.data.schema import POLARITY_CLASSES  # type: ignore


def probabilities(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def summarise(signed: np.ndarray, magnitude: np.ndarray, human: np.ndarray) -> dict:
    """What the sign does on pairs no one labelled for polarity."""
    finite = np.isfinite(human)
    negative = signed < 0
    return {
        "n": int(len(signed)),
        "share_negative": float(negative.mean()),
        "human_mean_on_negative": float(human[negative & finite].mean()) if (negative & finite).any() else float("nan"),
        "human_mean_on_positive": (
            float(human[~negative & finite].mean()) if (~negative & finite).any() else float("nan")
        ),
        "pearson_signed": float(pearsonr(signed[finite], human[finite])[0]) if finite.sum() > 2 else float("nan"),
        "pearson_magnitude": float(pearsonr(magnitude[finite], human[finite])[0]) if finite.sum() > 2 else float("nan"),
        "signed_min": float(signed.min()),
        "signed_median": float(np.median(signed)),
    }


@click.command()
@click.option("--magnitude", default="davebulaval/MeaningBERT", show_default=True)
@click.option("--magnitude-subfolder", default="large", show_default=True)
@click.option("--polarity", required=True)
@click.option("--dataset", default="davebulaval/CSMD", show_default=True)
@click.option("--subset", default="meaning", show_default=True)
@click.option("--split", default="test", show_default=True)
@click.option("--alpha", default=2.0, show_default=True)
@click.option("--json-out", default=None)
def main(
    magnitude: str,
    magnitude_subfolder: str,
    polarity: str,
    dataset: str,
    subset: str,
    split: str,
    alpha: float,
    json_out: Optional[str],
) -> None:
    """Score a simplification corpus with the composed signed scale."""
    from datasets import load_dataset

    rows = load_dataset(dataset, subset)[split]
    left = list(rows["original"])
    right = list(rows["simplification"])
    human = np.array(rows["label"], dtype=float)
    click.echo(f"{dataset}/{subset}/{split} : {len(left)} paires")

    magnitude_model = Model(magnitude, magnitude_subfolder or None)
    scores = magnitude_model.meaning_score(magnitude_model.logits(left, right))
    del magnitude_model

    polarity_model = Model(polarity, None)
    p_contradiction = probabilities(polarity_model.logits(left, right))[:, POLARITY_CLASSES["contradiction"]]
    del polarity_model

    signed = compose(scores, p_contradiction, alpha)
    got = summarise(signed, scores, human)
    got.update({"alpha": alpha, "dataset": f"{dataset}/{subset}/{split}", "polarity_checkpoint": polarity})

    click.echo(f"part de paires sous zero  : {100 * got['share_negative']:.2f}%")
    click.echo(f"note humaine moyenne, negatives : {got['human_mean_on_negative']:.1f}")
    click.echo(f"note humaine moyenne, positives : {got['human_mean_on_positive']:.1f}")
    click.echo(f"Pearson, score signe      : {got['pearson_signed']:.4f}")
    click.echo(f"Pearson, magnitude seule  : {got['pearson_magnitude']:.4f}")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump({**got, "signed": [float(v) for v in signed], "human": [float(v) for v in human]}, handle)
        click.echo(f"brut : {json_out}")


if __name__ == "__main__":
    main()
