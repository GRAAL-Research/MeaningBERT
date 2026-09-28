"""Compose the two heads into one signed score, and calibrate the form on SICK.

Experiment 4 of ``docs/v3-echelle-signee.md``. The v2 head answers how much meaning two
sentences share, from 0 to 100. The v3 head answers whether the second follows from the
first or denies it. Neither alone is the metric: a pair like "je bois du lait" against "je
ne bois pas du lait" shares almost every word and its subject, so the magnitude head says
90 and means "the meaning survived", which is the exact opposite of the truth.

The composition is a product, not a subtraction::

    signe = magnitude x (1 - alpha x p_contradiction)

Two sentences with nothing in common score 0 whatever their polarity, because contradicting
each other requires talking about the same thing first. A subtraction would let an
unrelated pair drift negative, which is the distinction the whole scale rests on.

**No retraining is involved.** This reads two existing checkpoints and combines their
outputs; the only thing fitted is ``alpha``, and it is fitted on a few thousand rows.

**One requirement is a constraint and not a trade.** The slope must let the score reach
-100, which happens from ``alpha = 2`` upwards. Below it the negative half is capped: at
1.5 a certain contradiction on a perfect magnitude lands at -50, and the scale is announced
as [-100, 100]. The v2 campaign paid for that lesson once already, when its sigmoid head
plateaued at 96.36 on identical pairs and had to be replaced. So reachability filters the
grid rather than joining the objective, because it is the definition of the scale and not
a quantity to weigh against the others.

**What calibration can and cannot mean here.** No corpus carries a human-annotated signed
score, so there is nothing to regress against; fitting to a fabricated target would measure
our own conversion rule. What SICK does give, uniquely, is both annotations on the same
pairs. So the form is chosen against three requirements at once, multiplied the way the v2
campaign's objective was, so that collapsing one of them cannot be hidden by the others:

* contradictions must land **below zero**;
* entailments must land **above zero**;
* on the pairs whose sign does not flip, the score must still track the human relatedness,
  because a composition that destroys the magnitude has bought the sign with the thing it
  was supposed to protect.

Run::

    PYTHONPATH=src python src/diagnostics/composition.py \\
        --magnitude davebulaval/MeaningBERT --magnitude-subfolder large \\
        --polarity results/polarity/nli-deberta-v3-base-none-poids/seed42/model
"""

from __future__ import annotations

import json
from typing import Any, Optional

import click
import numpy as np
from scipy.stats import pearsonr

try:  # PYTHONPATH=src.
    from data.schema import POLARITY_CLASSES
    from diagnostics.cross_task import Model
except ImportError:  # pragma: no cover
    from src.data.schema import POLARITY_CLASSES  # type: ignore
    from src.diagnostics.cross_task import Model  # type: ignore

#: The slope the document proposes: ``p = 0`` leaves the magnitude untouched, ``p = 0.5``
#: sends it to zero, ``p = 1`` flips it whole. Calibration exists to check that 2 is the
#: right number, not to assume it.
DEFAULT_ALPHA: float = 2.0

#: Slopes to sweep. Below 1 the score can never go negative, which defeats the purpose;
#: above 3 a merely probable contradiction is enough to invert a confident magnitude.
ALPHA_GRID: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0)


def scale_floor(alpha: float) -> float:
    """The most negative score this slope can ever produce.

    A certain contradiction on a pair whose magnitude is 100 lands at
    ``100 x (1 - alpha)``. So the negative half of the scale is only usable end to end from
    ``alpha = 2`` upwards; below it the scale is one-eyed. At 1.5 the floor is -50, and the
    document defines -100 as "sens oppose", the score that "je bois du lait" against "je ne
    bois pas du lait" is supposed to receive.

    Args:
        alpha: The slope.

    Returns:
        The floor, clipped to the declared range.
    """
    return max(100.0 * (1.0 - alpha), -100.0)


def reaches_full_scale(alpha: float) -> bool:
    """Whether *alpha* lets the score reach -100.

    Decision of David, 2026-09-28. The calibration, left to the three original
    requirements, preferred 1.5 by 1.7 percent relative, and 1.5 caps the negative half at
    -50. That is the defect the v2 campaign already paid for once: its sigmoid head
    plateaued at 96.36 on identical pairs and was replaced by a bounded one precisely so
    the endpoints could be reached. Publishing a scale announced as [-100, 100] whose lower
    third is never used would repeat it on the other side.

    So reachability is a constraint on the grid rather than a term added to the objective:
    it is not a quantity to trade against the others, it is the definition of the scale.
    """
    return scale_floor(alpha) <= -100.0


def compose(magnitude: np.ndarray, p_contradiction: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Combine a 0-100 magnitude and a contradiction probability into ``[-100, 100]``.

    Args:
        magnitude: Meaning-preservation scores, 0 to 100.
        p_contradiction: Probability the pair contradicts, 0 to 1.
        alpha: Slope. See :data:`DEFAULT_ALPHA`.

    Returns:
        Signed scores, clipped to the declared range.
    """
    return np.clip(magnitude * (1.0 - alpha * p_contradiction), -100.0, 100.0)


def sign_rates(signed: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Share of each class that lands on the side of zero it should.

    Neutral is reported but not required: a neutral pair shares its subject without
    asserting or denying, so any small score is defensible and demanding a side would be
    inventing a requirement the annotation does not support.
    """
    rates = {}
    for name, index in POLARITY_CLASSES.items():
        rows = signed[truth == index]
        if not len(rows):
            rates[name] = float("nan")
            continue
        rates[name] = float((rows < 0).mean() if name == "contradiction" else (rows > 0).mean())
    return rates


def magnitude_preserved(signed: np.ndarray, relatedness: np.ndarray, truth: np.ndarray) -> float:
    """Correlation with human relatedness on the pairs whose sign does not flip.

    A composition that sends every contradiction to -100 and everything else to noise would
    score perfectly on the sign and be useless. This is the counterweight.
    """
    keep = truth != POLARITY_CLASSES["contradiction"]
    if keep.sum() < 3:
        return float("nan")
    return float(pearsonr(signed[keep], relatedness[keep])[0])


def objective(signed: np.ndarray, relatedness: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """The three requirements and their product.

    A product and not a mean, for the reason the v2 campaign settled on one: a sum lets a
    composition trade the sign away for correlation, or the reverse, and report a
    respectable number while failing at the thing it exists for.
    """
    rates = sign_rates(signed, truth)
    preserved = magnitude_preserved(signed, relatedness, truth)
    score = rates["contradiction"] * rates["entailment"] * max(preserved, 0.0)
    return {
        "contradictions_negatives": rates["contradiction"],
        "implications_positives": rates["entailment"],
        "neutres_positifs": rates["neutral"],
        "pearson_proximite": preserved,
        "objectif": float(score),
    }


def calibrate(
    magnitude: np.ndarray,
    p_contradiction: np.ndarray,
    relatedness: np.ndarray,
    truth: np.ndarray,
    grid: tuple[float, ...] = ALPHA_GRID,
    require_full_scale: bool = True,
) -> tuple[float, list[dict[str, Any]]]:
    """Pick the slope that satisfies the three requirements best, among the usable ones.

    Args:
        magnitude: Meaning-preservation scores.
        p_contradiction: Contradiction probabilities.
        relatedness: Human relatedness, for the counterweight.
        truth: Polarity class indices.
        grid: Slopes to try.
        require_full_scale: Keep only the slopes that can reach -100. See
            :func:`reaches_full_scale` for why this is a constraint and not a fourth term.

    Returns:
        The winning alpha and the WHOLE curve, rejected slopes included, so the choice can
        be argued with rather than taken on trust.

    Raises:
        ValueError: If no slope in the grid can reach the full scale.
    """
    curve = []
    for alpha in grid:
        got = objective(compose(magnitude, p_contradiction, alpha), relatedness, truth)
        got["alpha"] = alpha
        got["plancher"] = scale_floor(alpha)
        got["echelle_complete"] = reaches_full_scale(alpha)
        curve.append(got)

    eligible = [row for row in curve if row["echelle_complete"]] if require_full_scale else curve
    if not eligible:
        raise ValueError(
            f"aucune pente de {grid} n'atteint -100 ; la moitie negative de l'echelle serait inutilisable"
        )
    best = max(eligible, key=lambda row: (row["objectif"], -row["alpha"]))
    return float(best["alpha"]), curve


def _probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax over the class axis."""
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


@click.command()
@click.option("--magnitude", required=True, help="Checkpoint of the v2 meaning-preservation head.")
@click.option("--magnitude-subfolder", default="large", show_default=True)
@click.option("--polarity", required=True, help="Checkpoint of the v3 polarity head.")
@click.option("--corpus", default="datastore/polarity/corpus", show_default=True)
@click.option("--json-out", default=None)
def main(magnitude: str, magnitude_subfolder: str, polarity: str, corpus: str, json_out: Optional[str]) -> None:
    """Calibrate the composition on the SICK half of the v3 test split."""
    from datasets import load_from_disk

    test = load_from_disk(corpus)["test"]
    # SICK only: it is the one corpus carrying a relatedness score AND an inference label on
    # the same pairs, which is the only thing that makes this calibration possible at all.
    bridge = test.filter(lambda row: row["corpus"] == "sick")
    left, right = list(bridge["original"]), list(bridge["simplification"])
    truth = np.array(bridge["polarity"], dtype=float).astype(int)
    # SICK relatedness is a 1-5 Likert; the scale itself does not matter to a correlation,
    # only its ordering, so it is used as published.
    relatedness = np.array(bridge["label_raw"], dtype=float)
    click.echo(f"pont SICK : {len(bridge)} paires, dont {int((truth == POLARITY_CLASSES['contradiction']).sum())} contradictions\n")

    magnitude_model = Model(magnitude, magnitude_subfolder or None)
    if magnitude_model.is_polarity:
        raise click.ClickException(f"{magnitude} est une tete de polarite, pas une tete de magnitude")
    scores = magnitude_model.meaning_score(magnitude_model.logits(left, right))
    del magnitude_model

    polarity_model = Model(polarity, None)
    if not polarity_model.is_polarity:
        raise click.ClickException(f"{polarity} n'est pas une tete a trois classes")
    p_contradiction = _probabilities(polarity_model.logits(left, right))[:, POLARITY_CLASSES["contradiction"]]
    del polarity_model

    best, curve = calibrate(scores, p_contradiction, relatedness, truth)
    click.echo(
        "%6s %14s %14s %12s %10s %9s"
        % ("alpha", "contra < 0", "implic. > 0", "Pearson prox.", "objectif", "plancher")
    )
    for row in curve:
        mark = " <-" if row["alpha"] == best else ("" if row["echelle_complete"] else "  (echelle borgne)")
        click.echo(
            "%6.2f %14.4f %14.4f %12.4f %10.4f %9.0f%s"
            % (row["alpha"], row["contradictions_negatives"], row["implications_positives"],
               row["pearson_proximite"], row["objectif"], row["plancher"], mark)
        )

    signed = compose(scores, p_contradiction, best)
    click.echo(f"\nalpha retenu : {best}")
    click.echo(f"score signe : min {signed.min():.1f}  median {np.median(signed):.1f}  max {signed.max():.1f}")
    magnitude_only = objective(scores, relatedness, truth)
    click.echo(
        "sans composition, la magnitude seule : contradictions negatives "
        f"{magnitude_only['contradictions_negatives']:.4f}, objectif {magnitude_only['objectif']:.4f}"
    )

    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {"alpha": best, "curve": curve, "magnitude_only": magnitude_only,
                 "magnitude_checkpoint": magnitude, "polarity_checkpoint": polarity},
                handle, indent=2,
            )
        click.echo(f"brut : {json_out}")


if __name__ == "__main__":
    main()
