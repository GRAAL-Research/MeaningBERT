"""Tests for the composition figures (``src/figures_generator/figures_composition.py``).

Both figures carry a claim the article makes in words, so a wrong curve argues for the
wrong conclusion without any step failing. The tests pin the two computations the eye
cannot check: the binning that normalises three classes of different sizes onto one axis,
and the reliability curve that the article reads as evidence of miscalibration.
"""

from __future__ import annotations

import json

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
        assert r"\textcolor{centailment}{\textbf{entailment}} ($n = 3$)" in caption
        assert r"\textcolor{ccontradiction}{\textbf{contradiction}} ($n = 2$)" in caption

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
                    "pearson_proximite": 0.83,
                    "objectif": 0.0,
                    "plancher": 0.0,
                },
                {
                    "alpha": 2.00,
                    "contradictions_negatives": 0.837,
                    "implications_positives": 1.0,
                    "pearson_proximite": 0.795,
                    "objectif": 0.665,
                    "plancher": -100.0,
                },
            ],
            "magnitude_only": {
                "contradictions_negatives": 0.0,
                "implications_positives": 1.0,
                "pearson_proximite": 0.852,
            },
        }

        composition_table(curve, str(path))
        body = path.read_text(encoding="utf-8")

        assert r"\textbf{83.7}" in body
        assert r"\textbf{0.0}" not in body
        assert "Magnitude alone & 0.0" in body

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
                    "pearson_proximite": 0.7,
                    "objectif": 0.4,
                    "plancher": -100.0,
                }
            ],
            "magnitude_only": {
                "contradictions_negatives": 0.0,
                "implications_positives": 1.0,
                "pearson_proximite": 0.8,
            },
        }

        composition_table(curve, str(path))

        assert json.dumps(path.read_text(encoding="utf-8")).count("alpha = 1.25") == 0


class TestDecisionPoint:
    """The measurement that turns 'expresses a distinction' into 'is a better metric'."""

    def test_the_signed_scale_rejects_the_contradictions_the_magnitude_accepts(self):
        from figures_generator.figures_composition import decision_point

        # Two contradictions a magnitude metric waves through, two entailments it should keep.
        pairs = {
            "magnitude": [80.0, 75.0, 90.0, 85.0],
            "signed": [-40.0, -35.0, 89.0, 84.0],
            "truth": [POLARITY_CLASSES["contradiction"]] * 2 + [POLARITY_CLASSES["entailment"]] * 2,
        }

        (row,) = decision_point(pairs, thresholds=(50,))

        assert row["magnitude"]["share_contradiction"] == pytest.approx(0.5)
        assert row["signed"]["share_contradiction"] == pytest.approx(0.0)
        assert row["signed"]["entailment_recall"] == pytest.approx(1.0)

    def test_a_scale_that_only_shifts_everything_down_is_caught_by_the_control(self):
        """Rejecting contradictions by rejecting everything must show up as lost recall."""
        from figures_generator.figures_composition import decision_point

        pairs = {
            "magnitude": [80.0, 90.0],
            "signed": [-10.0, -10.0],
            "truth": [POLARITY_CLASSES["contradiction"], POLARITY_CLASSES["entailment"]],
        }

        (row,) = decision_point(pairs, thresholds=(50,))

        assert row["signed"]["share_contradiction"] != row["signed"]["share_contradiction"]  # NaN
        assert row["signed"]["entailment_recall"] == pytest.approx(0.0)

    def test_an_empty_acceptance_set_gives_nan_rather_than_a_perfect_score(self):
        """Accepting nothing is not a metric with zero false acceptances."""
        from figures_generator.figures_composition import decision_point

        pairs = {"magnitude": [10.0], "signed": [10.0], "truth": [POLARITY_CLASSES["contradiction"]]}

        (row,) = decision_point(pairs, thresholds=(50,))

        assert row["magnitude"]["share_contradiction"] != row["magnitude"]["share_contradiction"]

    def test_the_table_bolds_the_quantity_the_paper_argues_about(self, tmp_path):
        from figures_generator.figures_composition import decision_table

        pairs = {
            "magnitude": [80.0, 90.0],
            "signed": [-40.0, 89.0],
            "truth": [POLARITY_CLASSES["contradiction"], POLARITY_CLASSES["entailment"]],
        }
        path = tmp_path / "t.tex"
        decision_table(pairs, str(path))
        body = path.read_text(encoding="utf-8")

        assert r"\textbf{0.00}" in body
        assert "50.00" in body
