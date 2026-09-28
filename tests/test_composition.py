"""Tests for the composition of the two heads (``src/diagnostics/composition.py``).

This is where the signed score is actually produced, so the tests are about the three
properties the scale is defined by, and about the calibration refusing to buy one of them
with another.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from data.schema import POLARITY_CLASSES
from diagnostics.composition import (
    ALPHA_GRID,
    DEFAULT_ALPHA,
    calibrate,
    compose,
    magnitude_preserved,
    objective,
    sign_rates,
)

ENTAIL = POLARITY_CLASSES["entailment"]
NEUTRAL = POLARITY_CLASSES["neutral"]
CONTRA = POLARITY_CLASSES["contradiction"]


def _c(magnitude, p, alpha=DEFAULT_ALPHA):
    return compose(np.array([magnitude], dtype=float), np.array([p], dtype=float), alpha)[0]


# --- the three worked examples of the scale ------------------------------------------


def test_a_paraphrase_keeps_its_magnitude():
    # "je bois du lait" / "je consomme du lait"
    assert _c(85.0, 0.02) == pytest.approx(81.6)


def test_a_negation_of_the_same_sentence_flips_the_sign():
    # "je bois du lait" / "je ne bois pas du lait": nearly every word shared, so the
    # magnitude head alone says 90 and means "the meaning survived".
    assert _c(90.0, 0.97) < -80.0


def test_an_unrelated_pair_stays_near_zero_whatever_its_polarity():
    # The distinction the whole scale rests on: contradicting requires talking about the
    # same thing first. A subtraction would let this drift negative.
    assert _c(3.0, 0.10) == pytest.approx(2.4)
    assert abs(_c(3.0, 0.99)) < 3.0
    assert abs(_c(0.0, 1.0)) == pytest.approx(0.0)


def test_a_certain_contradiction_inverts_the_magnitude_whole():
    assert _c(100.0, 1.0) == pytest.approx(-100.0)


def test_an_even_chance_of_contradiction_lands_on_zero():
    assert _c(100.0, 0.5) == pytest.approx(0.0)


def test_the_score_never_leaves_the_declared_range():
    assert _c(100.0, 1.0, alpha=5.0) == pytest.approx(-100.0)
    assert _c(100.0, 0.0, alpha=5.0) == pytest.approx(100.0)


def test_the_default_slope_is_the_one_the_document_defines():
    assert DEFAULT_ALPHA == 2.0


# --- the three requirements -----------------------------------------------------------


def test_sign_rates_count_each_class_on_the_side_it_belongs():
    signed = np.array([50.0, -30.0, -10.0, 20.0])
    truth = np.array([ENTAIL, CONTRA, ENTAIL, CONTRA])
    rates = sign_rates(signed, truth)
    assert rates["entailment"] == pytest.approx(0.5)
    assert rates["contradiction"] == pytest.approx(0.5)


def test_a_class_absent_from_the_data_reports_nan_rather_than_zero():
    rates = sign_rates(np.array([10.0]), np.array([ENTAIL]))
    assert math.isnan(rates["contradiction"])


def test_neutral_is_measured_but_never_required():
    # A neutral pair shares its subject without asserting or denying, so demanding a side
    # would invent a requirement the annotation does not support.
    signed = np.array([10.0, -10.0])
    truth = np.array([NEUTRAL, NEUTRAL])
    got = objective(signed, np.array([3.0, 3.0]), truth)
    assert math.isnan(got["objectif"]) or got["neutres_positifs"] == pytest.approx(0.5)


def test_the_magnitude_counterweight_ignores_the_pairs_whose_sign_flipped():
    # The contradiction row is wildly out of order with its relatedness; including it would
    # drag the correlation down for doing exactly what the composition is supposed to do.
    signed = np.array([80.0, 60.0, 40.0, 20.0, -90.0])
    relatedness = np.array([5.0, 4.0, 3.0, 2.0, 4.5])
    truth = np.array([ENTAIL, NEUTRAL, NEUTRAL, ENTAIL, CONTRA])
    assert magnitude_preserved(signed, relatedness, truth) == pytest.approx(1.0)


def test_a_correlation_on_two_points_is_refused_rather_than_reported_as_perfect():
    # Two points always correlate at plus or minus one, so the number would be a fact about
    # the sample size and not about the model.
    signed = np.array([80.0, 40.0, -90.0])
    relatedness = np.array([5.0, 3.0, 4.5])
    truth = np.array([ENTAIL, NEUTRAL, CONTRA])
    assert math.isnan(magnitude_preserved(signed, relatedness, truth))


# --- calibration ----------------------------------------------------------------------


def _bridge(alpha_truth=2.0, n=60):
    """A synthetic SICK-like bridge whose ideal slope is known."""
    rng = np.random.default_rng(0)
    relatedness = rng.uniform(1.0, 5.0, n)
    magnitude = relatedness * 20.0
    truth = np.array([CONTRA if i % 3 == 0 else (ENTAIL if i % 3 == 1 else NEUTRAL) for i in range(n)])
    p = np.where(truth == CONTRA, 0.95, 0.02)
    del alpha_truth
    return magnitude, p, relatedness, truth


def test_calibration_returns_a_slope_from_the_grid_and_the_whole_curve():
    best, curve = calibrate(*_bridge())
    assert best in ALPHA_GRID
    assert len(curve) == len(ALPHA_GRID)
    assert {"alpha", "objectif", "pearson_proximite"} <= set(curve[0])


def test_a_slope_too_small_to_ever_go_negative_is_rejected():
    # At alpha = 1 the score is magnitude * (1 - p), which cannot be negative, so no
    # contradiction ever lands where it belongs.
    magnitude, p, relatedness, truth = _bridge()
    at_one = next(row for row in calibrate(magnitude, p, relatedness, truth)[1] if row["alpha"] == 1.0)
    assert at_one["contradictions_negatives"] == pytest.approx(0.0)
    assert at_one["objectif"] == pytest.approx(0.0)


def test_the_chosen_slope_gets_the_contradictions_on_the_right_side():
    magnitude, p, relatedness, truth = _bridge()
    best, _ = calibrate(magnitude, p, relatedness, truth)
    signed = compose(magnitude, p, best)
    assert sign_rates(signed, truth)["contradiction"] == pytest.approx(1.0)


def test_the_objective_is_a_product_so_one_collapse_sinks_it():
    # The reason it is not a mean: a composition must not be able to trade the sign away
    # for correlation and still report a respectable number.
    signed = np.array([80.0, 60.0, 40.0, 20.0, 90.0])
    relatedness = np.array([5.0, 4.0, 3.0, 2.0, 4.5])
    truth = np.array([ENTAIL, NEUTRAL, NEUTRAL, ENTAIL, CONTRA])
    got = objective(signed, relatedness, truth)
    assert got["contradictions_negatives"] == pytest.approx(0.0)
    assert got["objectif"] == pytest.approx(0.0)
    assert got["pearson_proximite"] > 0.9


def test_a_shorter_slope_wins_ties():
    # Two slopes that satisfy the requirements equally: prefer the one that disturbs the
    # magnitude least, since the magnitude carries the human annotation.
    magnitude, p, relatedness, truth = _bridge()
    best, curve = calibrate(magnitude, p, relatedness, truth)
    # Among the slopes that keep the scale whole: the rejected ones are still in the curve,
    # but they were never candidates.
    eligible = [row for row in curve if row["echelle_complete"]]
    winners = [row["alpha"] for row in eligible if row["objectif"] == pytest.approx(max(r["objectif"] for r in eligible))]
    assert best == min(winners)


# --- reachability of the scale, a constraint and not a trade -------------------------

from diagnostics.composition import reaches_full_scale, scale_floor  # noqa: E402


def test_the_floor_is_what_a_certain_contradiction_can_reach():
    # 100 x (1 - alpha), which is the whole arithmetic of the question.
    assert scale_floor(1.5) == pytest.approx(-50.0)
    assert scale_floor(2.0) == pytest.approx(-100.0)


def test_a_slope_below_two_leaves_the_negative_half_unusable():
    # At 1.5 a certain contradiction on a perfect magnitude lands at -50, so a scale
    # announced as [-100, 100] never uses its lower third.
    assert not reaches_full_scale(1.5)
    assert not reaches_full_scale(1.99)


def test_two_is_exactly_where_the_scale_becomes_whole():
    assert reaches_full_scale(2.0)
    assert reaches_full_scale(3.0)


def test_a_steeper_slope_cannot_push_the_score_past_the_declared_range():
    assert scale_floor(5.0) == pytest.approx(-100.0)


def test_calibration_refuses_a_slope_that_truncates_the_scale():
    # The decision of 2026-09-28: the three requirements preferred 1.5, and 1.5 caps the
    # negative half. Reachability is the definition of the scale, not a term to weigh
    # against the rest, so it filters the grid.
    magnitude, p, relatedness, truth = _bridge()
    best, curve = calibrate(magnitude, p, relatedness, truth)
    assert best >= 2.0
    assert reaches_full_scale(best)
    # The rejected slopes stay in the curve so the choice can be argued with.
    assert any(not row["echelle_complete"] for row in curve)


def test_the_curve_reports_the_floor_of_every_slope():
    _, curve = calibrate(*_bridge())
    assert all("plancher" in row and "echelle_complete" in row for row in curve)
    assert next(row for row in curve if row["alpha"] == 1.5)["plancher"] == pytest.approx(-50.0)


def test_the_constraint_can_be_lifted_explicitly():
    # Kept switchable so the cost of the decision stays measurable, not to be used by
    # default.
    magnitude, p, relatedness, truth = _bridge()
    free, _ = calibrate(magnitude, p, relatedness, truth, require_full_scale=False)
    constrained, _ = calibrate(magnitude, p, relatedness, truth)
    assert free <= constrained


def test_a_grid_with_no_usable_slope_is_refused_rather_than_silently_truncating():
    magnitude, p, relatedness, truth = _bridge()
    with pytest.raises(ValueError, match="inutilisable"):
        calibrate(magnitude, p, relatedness, truth, grid=(1.0, 1.5), require_full_scale=True)
