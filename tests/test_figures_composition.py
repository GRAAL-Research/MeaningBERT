"""Tests for the composition figures (``src/figures_generator/figures_composition.py``).

Both figures carry a claim the article makes in words, so a wrong curve argues for the
wrong conclusion without any step failing. The tests pin the two computations the eye
cannot check: the binning that normalises three classes of different sizes onto one axis,
and the reliability curve that the article reads as evidence of miscalibration.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from data.schema import POLARITY_CLASSES
from figures_generator.figures_composition import (
    composition_table,
    distribution_figure,
    histogram,
    reliability,
    reliability_figure,
)


class TestHistogram:
    def test_each_class_is_normalised_to_its_own_total(self):
        """Classes differ in size by a factor of three; raw counts would hide the shape."""
        edges = np.array([0.0, 1.0, 2.0])

        got = histogram(np.array([0.5, 0.5, 1.5]), edges)

        assert got.sum() == pytest.approx(1.0)
        assert got[0] == pytest.approx(2 / 3)

    def test_an_empty_class_gives_zeros_rather_than_dividing_by_zero(self):
        got = histogram(np.array([]), np.array([0.0, 1.0, 2.0]))

        assert got.tolist() == [0.0, 0.0]


class TestReliability:
    def test_a_calibrated_head_follows_the_diagonal(self):
        """Half the pairs at probability 0.5 really are contradictions."""
        probability = np.array([0.05] * 20 + [0.55] * 20 + [0.95] * 20)
        observed = np.array([0.0] * 19 + [1.0] + [0.0] * 10 + [1.0] * 10 + [0.0] + [1.0] * 19)

        curve = reliability(probability, observed)

        for predicted, rate, _ in curve:
            assert abs(predicted - rate) < 0.1

    def test_an_over_asserting_head_falls_below_the_diagonal(self):
        """The article reads this gap as miscalibration, so the sign of it must be right."""
        probability = np.array([0.55] * 20)
        observed = np.array([1.0] + [0.0] * 19)

        ((predicted, rate, count),) = reliability(probability, observed)

        assert rate < predicted
        assert count == 20

    def test_an_empty_bin_is_dropped_rather_than_plotted_at_zero(self):
        """A bin no pair falls into is not a bin where the head is always wrong."""
        curve = reliability(np.array([0.95, 0.96]), np.array([1.0, 1.0]))

        assert len(curve) == 1

    def test_the_last_bin_includes_a_probability_of_exactly_one(self):
        """A saturated head puts mass on 1.0, and a half-open bin would drop it."""
        curve = reliability(np.array([1.0, 1.0]), np.array([1.0, 0.0]))

        assert sum(count for _, _, count in curve) == 2


def pairs_fixture() -> dict:
    truth = [POLARITY_CLASSES["entailment"]] * 3 + [POLARITY_CLASSES["contradiction"]] * 2
    return {
        "signed": [90.0, 85.0, 80.0, -40.0, -55.0],
        "p_contradiction": [0.01, 0.02, 0.03, 0.9, 0.95],
        "truth": truth,
        "relatedness": [4.5, 4.4, 4.3, 3.0, 2.9],
        "alpha": 2.0,
    }


class TestFigureOutput:
    def test_the_classes_are_named_in_the_caption_and_not_in_the_panel(self, tmp_path):
        """Three legend entries took a quarter of the panel and covered the left peak."""
        path = tmp_path / "d.tex"

        distribution_figure(pairs_fixture(), str(path))
        body = path.read_text(encoding="utf-8")

        assert r"\addlegendentry" not in body and "legend style" not in body
        caption = body.split(r"\caption{")[1]
        assert r"\textcolor{centailment}{\textbf{entailment}}" in caption
        assert r"\textcolor{ccontradiction}{\textbf{contradiction}}" in caption

    def test_the_reliability_curves_are_named_in_the_caption_in_their_own_colour(self, tmp_path):
        """Colour is the only thing telling the two curves apart, so it must travel with the name."""
        path = tmp_path / "r.tex"

        reliability_figure(pairs_fixture(), pairs_fixture(), str(path))
        body = path.read_text(encoding="utf-8")

        assert r"\addlegendentry" not in body and "legend style" not in body
        assert "forget plot" in body
        assert r"\textcolor{reltuned}{\textbf{fine-tuned head}}" in body
        assert r"\textcolor{relshelf}{\textbf{off-the-shelf head}}" in body

    def test_the_composition_table_bolds_the_slope_that_was_fitted(self, tmp_path):
        path = tmp_path / "t.tex"
        curve = {
            "alpha": 2.0,
            "curve": [
                {
                    "alpha": 1.00,
                    "contradictions_negatives": 0.0,
                    "implications_positives": 1.0,
                    "neutres_positifs": 0.9835,
                    "pearson_proximite": 0.83,
                    "objectif": 0.0,
                    "plancher": 0.0,
                },
                {
                    "alpha": 2.00,
                    "contradictions_negatives": 0.837,
                    "implications_positives": 1.0,
                    "neutres_positifs": 0.9745,
                    "pearson_proximite": 0.795,
                    "objectif": 0.665,
                    "plancher": -100.0,
                },
            ],
            "magnitude_only": {
                "contradictions_negatives": 0.0,
                "implications_positives": 1.0,
                "neutres_positifs": 0.9835,
                "pearson_proximite": 0.852,
            },
        }

        composition_table(curve, str(path))
        body = path.read_text(encoding="utf-8")

        assert r"\textbf{83.70}" in body
        assert r"\textbf{97.45}" in body
        assert r"\textbf{0.00}" not in body
        assert "Magnitude alone & 0.00" in body

    def test_the_table_survives_a_curve_missing_a_slope(self, tmp_path):
        """A shorter grid should drop rows, not raise on the way to the paper."""
        path = tmp_path / "t.tex"
        curve = {
            "alpha": 2.0,
            "curve": [
                {
                    "alpha": 2.00,
                    "contradictions_negatives": 0.5,
                    "implications_positives": 1.0,
                    "neutres_positifs": 0.97,
                    "pearson_proximite": 0.7,
                    "objectif": 0.4,
                    "plancher": -100.0,
                }
            ],
            "magnitude_only": {
                "contradictions_negatives": 0.0,
                "implications_positives": 1.0,
                "neutres_positifs": 0.98,
                "pearson_proximite": 0.8,
            },
        }

        composition_table(curve, str(path))

        assert json.dumps(path.read_text(encoding="utf-8")).count("alpha = 1.25") == 0


class TestDecisionPoint:
    """The measurement that turns 'expresses a distinction' into 'is a better metric'."""

    def test_the_signed_scale_rejects_the_contradictions_the_magnitude_accepts(self):
        from figures_generator.figures_composition import accepted_rates

        # Two contradictions a magnitude metric waves through, two entailments it keeps.
        truth = np.array([POLARITY_CLASSES["contradiction"]] * 2 + [POLARITY_CLASSES["entailment"]] * 2)
        magnitude = np.array([80.0, 75.0, 90.0, 85.0])
        signed = np.array([-40.0, -35.0, 89.0, 84.0])

        assert accepted_rates(magnitude, truth, 50)[0] == pytest.approx(0.5)
        assert accepted_rates(signed, truth, 50)[0] == pytest.approx(0.0)
        assert accepted_rates(signed, truth, 50)[1] == pytest.approx(1.0)

    def test_a_scale_that_only_shifts_everything_down_is_caught_by_the_control(self):
        """Rejecting contradictions by rejecting everything must show up as lost recall."""
        from figures_generator.figures_composition import accepted_rates

        truth = np.array([POLARITY_CLASSES["contradiction"], POLARITY_CLASSES["entailment"]])
        share, recall = accepted_rates(np.array([-10.0, -10.0]), truth, 50)

        assert math.isnan(share)
        assert recall == pytest.approx(0.0)

    def test_an_empty_acceptance_set_gives_nan_rather_than_a_perfect_score(self):
        """Accepting nothing is not a metric with zero false acceptances."""
        from figures_generator.figures_composition import accepted_rates

        truth = np.array([POLARITY_CLASSES["contradiction"]])
        assert math.isnan(accepted_rates(np.array([10.0]), truth, 50)[0])

    def test_the_bootstrap_brackets_the_point_estimate(self):
        """An interval that does not contain the value it describes is worse than none."""
        from figures_generator.figures_composition import accepted_rates, bootstrap_interval

        rng = np.random.default_rng(0)
        truth = np.array([POLARITY_CLASSES["contradiction"]] * 30 + [POLARITY_CLASSES["entailment"]] * 70)
        score = np.concatenate([rng.uniform(60, 90, 30), rng.uniform(60, 90, 70)])

        point = accepted_rates(score, truth, 50)[0]
        low, high = bootstrap_interval(score, truth, 50, draws=200)

        assert low <= point <= high
        assert high - low > 0

    def test_a_degenerate_scale_gives_a_zero_width_interval_rather_than_an_error(self):
        from figures_generator.figures_composition import bootstrap_interval

        truth = np.array([POLARITY_CLASSES["contradiction"]] * 5)
        low, high = bootstrap_interval(np.full(5, 90.0), truth, 50, draws=50)

        assert low == pytest.approx(1.0) and high == pytest.approx(1.0)

    def test_the_table_carries_the_three_scales_and_both_blocks(self, tmp_path):
        from figures_generator.figures_composition import decision_table

        truth = [POLARITY_CLASSES["contradiction"], POLARITY_CLASSES["entailment"]]
        tuned = {"truth": truth, "magnitude": [80.0, 90.0], "signed": [-40.0, 89.0]}
        shelf = {"truth": truth, "signed": [-70.0, 60.0]}
        path = tmp_path / "t.tex"

        decision_table(tuned, shelf, str(path), thresholds=(50,))
        body = path.read_text(encoding="utf-8")

        assert "Contradictions accepted" in body and "Entailments kept" in body
        for name in ("Magnitude alone", "Off-the-shelf head", "Ours, \\textsc{raw}"):
            assert body.count(name) == 2
        assert "bootstrap" in body


class TestSeveralHeads:
    """Section 7 rests on ten heads; one lucky seed must not carry the table."""

    @staticmethod
    def _curve(share: float, pearson: float) -> dict:
        row = {
            "alpha": 2.00,
            "contradictions_negatives": share,
            "implications_positives": 1.0,
            "neutres_positifs": 0.97,
            "pearson_proximite": pearson,
            "objectif": 0.6,
            "plancher": -100.0,
        }
        only = {"contradictions_negatives": 0.0, "neutres_positifs": 0.98, "pearson_proximite": 0.85}
        return {"alpha": 2.0, "curve": [row], "magnitude_only": only}

    def test_the_composition_table_reports_the_mean_and_spread_over_heads(self, tmp_path):
        path = tmp_path / "t.tex"

        composition_table([self._curve(0.80, 0.78), self._curve(0.90, 0.80)], str(path))
        body = path.read_text(encoding="utf-8")

        # mean 85.00, sample standard deviation of (80, 90) is 7.07
        assert r"\textbf{85.00$_{\pm 7.07}$}" in body
        assert r"\textbf{0.79$_{\pm 0.01}$}" in body
        assert "over 2 polarity heads" in body

    def test_the_fine_tuned_decision_rows_average_the_heads_not_the_first_one(self, tmp_path):
        from figures_generator.figures_composition import decision_table

        contra, entail = POLARITY_CLASSES["contradiction"], POLARITY_CLASSES["entailment"]
        truth = [contra, entail, entail, entail]
        first = {"truth": truth, "magnitude": [80.0, 90.0, 90.0, 90.0], "signed": [60.0, 89.0, 89.0, 89.0]}
        second = {"truth": truth, "magnitude": [80.0, 90.0, 90.0, 90.0], "signed": [-60.0, 89.0, 89.0, 89.0]}
        shelf = {"truth": truth, "signed": [-70.0, 60.0, 60.0, 60.0]}
        path = tmp_path / "t.tex"

        decision_table(first, shelf, str(path), thresholds=(50,), heads=[first, second])
        body = path.read_text(encoding="utf-8")

        # first head accepts the contradiction (1 of 4 accepted), second rejects it (0 of 3)
        assert r"Ours, \textsc{raw} & 12.50$_{\pm 17.68}$" in body
