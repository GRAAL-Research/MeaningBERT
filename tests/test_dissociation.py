"""Tests for the dissociation diagnostic (``src/diagnostics/dissociation.py``).

This is the measurement that produced the v3 baseline, so an error here does not merely
fail, it publishes a wrong number. The tests are concentrated on tie handling, because a
clamped output head saturates at exactly 0 and exactly 100 and ties between the two classes
are therefore the common case rather than an edge case.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from diagnostics.dissociation import amplitude


def _auc(entail: list[float], contra: list[float]) -> float:
    return amplitude(np.array(entail, dtype=float), np.array(contra, dtype=float))["auc"]


# --- amplitude, the headline number --------------------------------------------------


def test_amplitude_is_the_gap_between_the_two_class_means():
    got = amplitude(np.array([80.0, 60.0]), np.array([30.0, 10.0]))
    assert got["amplitude_points"] == pytest.approx(50.0)
    assert got["amplitude_share"] == pytest.approx(0.5)


def test_a_metric_that_scores_contradictions_higher_reports_a_negative_amplitude():
    # This is the published v1 result, -1.8 % on SICK, and it must not silently become a
    # positive number: a metric ranking contradictions above agreement is worse than one
    # that cannot tell, and the sign is the only thing that says so.
    got = amplitude(np.array([40.0]), np.array([60.0]))
    assert got["amplitude_points"] == pytest.approx(-20.0)


# --- AUC, and the ties that a clamped head guarantees --------------------------------


def test_perfect_separation_scores_one():
    assert _auc([90.0, 80.0], [20.0, 10.0]) == pytest.approx(1.0)


def test_perfectly_inverted_ranking_scores_zero():
    # The signature of a swapped label mapping. It is how the SICK encoding error was
    # caught on 2026-09-25, as an AUC of 0.021.
    assert _auc([10.0, 20.0], [80.0, 90.0]) == pytest.approx(0.0)


def test_two_classes_that_are_everywhere_identical_score_one_half():
    # The whole reason mid-ranks are used. With ordinal ranks this returned 0.0, which
    # reads as a perfectly inverted metric when the truth is that it cannot discriminate
    # at all.
    tied = [100.0] * 20
    assert _auc(tied, tied) == pytest.approx(0.5)


def test_the_auc_does_not_depend_on_which_class_is_passed_first():
    # With ordinal ranks, ties broke by position in the concatenated array, so swapping the
    # arguments moved the answer from 0.17 to 0.85 on the same data.
    entail = [100.0] * 50 + [80.0] * 10
    contra = [100.0] * 50 + [20.0] * 10
    assert _auc(entail, contra) == pytest.approx(1.0 - _auc(contra, entail))


def test_a_single_tied_pair_counts_as_half_a_correctly_ordered_couple():
    # Two couples: (90, 50) is ordered correctly, (50, 50) is a tie. Half credit gives
    # (1 + 0.5) / 2 = 0.75.
    assert _auc([90.0, 50.0], [50.0]) == pytest.approx(0.75)


def test_saturation_at_the_clamped_endpoints_is_scored_honestly():
    # What a clamped v2 checkpoint actually produces: most agreement pinned at 100, most
    # contradiction pinned at 0, and an overlapping tail. The tail is the only thing the
    # AUC can still resolve, and it must not be swamped by the tied mass.
    entail = [100.0] * 30 + [55.0] * 5
    contra = [0.0] * 30 + [55.0] * 5
    assert _auc(entail, contra) == pytest.approx((900 + 150 + 150 + 12.5) / 1225)


# --- the census that goes with the numbers -------------------------------------------


def test_the_class_counts_are_reported_beside_the_scores():
    got = amplitude(np.array([90.0, 80.0, 70.0]), np.array([10.0]))
    assert got["n_entailment"] == 3
    assert got["n_contradiction"] == 1


def test_an_empty_class_reports_nan_rather_than_a_confident_number():
    got = amplitude(np.array([90.0, 80.0]), np.array([]))
    assert math.isnan(got["auc"])
