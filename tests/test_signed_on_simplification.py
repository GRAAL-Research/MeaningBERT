"""Tests for the signed scale on simplification output.

``src/diagnostics/signed_on_simplification.py`` reports what the sign does on pairs no
one labelled for polarity, which means no assertion in it can be checked against a gold
label. The summary it produces is therefore the whole result, and these tests pin what
each of its fields must mean.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from diagnostics.signed_on_simplification import probabilities, summarise


class TestProbabilities:
    def test_each_row_sums_to_one(self):
        got = probabilities(np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]]))

        assert got.sum(axis=1) == pytest.approx([1.0, 1.0])

    def test_large_logits_do_not_overflow(self):
        """Subtracting the row maximum is what keeps a saturated head finite."""
        got = probabilities(np.array([[1000.0, 0.0, -1000.0]]))

        assert np.isfinite(got).all()
        assert got[0, 0] == pytest.approx(1.0)


class TestSummarise:
    def test_the_negative_share_counts_strictly_below_zero(self):
        """A score of exactly zero is an unrelated pair, not an opposed one."""
        signed = np.array([-1.0, 0.0, 1.0, 50.0])
        human = np.array([10.0, 20.0, 30.0, 40.0])

        assert summarise(signed, np.abs(signed), human)["share_negative"] == pytest.approx(0.25)

    def test_it_separates_the_human_ratings_of_the_two_sides(self):
        """The paper's claim is that the sign selects the pairs humans rated worst."""
        signed = np.array([-30.0, -20.0, 60.0, 70.0])
        human = np.array([10.0, 20.0, 80.0, 90.0])

        got = summarise(signed, np.abs(signed), human)

        assert got["human_mean_on_negative"] == pytest.approx(15.0)
        assert got["human_mean_on_positive"] == pytest.approx(85.0)

    def test_a_missing_human_rating_is_excluded_rather_than_counted_as_zero(self):
        signed = np.array([-10.0, -20.0, 50.0])
        human = np.array([float("nan"), 40.0, 90.0])

        got = summarise(signed, np.abs(signed), human)

        assert got["human_mean_on_negative"] == pytest.approx(40.0)

    def test_both_correlations_are_reported_so_the_cost_of_the_sign_is_visible(self):
        """The article compares them directly; reporting only one would hide the trade."""
        magnitude = np.array([10.0, 20.0, 30.0, 40.0])
        human = np.array([10.0, 20.0, 30.0, 40.0])
        signed = np.array([10.0, 20.0, 30.0, -40.0])

        got = summarise(signed, magnitude, human)

        assert got["pearson_magnitude"] == pytest.approx(1.0)
        assert got["pearson_signed"] < got["pearson_magnitude"]

    def test_no_negative_pair_gives_nan_rather_than_a_mean_of_nothing(self):
        signed = np.array([10.0, 20.0])
        got = summarise(signed, signed, np.array([50.0, 60.0]))

        assert got["share_negative"] == 0.0
        assert math.isnan(got["human_mean_on_negative"])
