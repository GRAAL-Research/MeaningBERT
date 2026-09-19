"""Tests for the training and evaluation metrics, including the C1 divergence flag."""

import math

import numpy as np
import pytest
from sklearn.metrics import root_mean_squared_error

from src.training.metrics.metrics import (
    DIVERGED_KEY,
    N_NON_FINITE_KEY,
    _sanitize_predictions,
    compute_metrics,
    eval_compute_metrics_identical,
    eval_compute_metrics_unrelated,
)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_metrics((predictions, labels))

        assert result["rmse"] == pytest.approx(0.0, abs=1e-6)
        r2_value = result["R2"]["r_squared"] if isinstance(result["R2"], dict) else result["R2"]
        assert r2_value == pytest.approx(1.0, abs=1e-6)
        assert result["pearson_corr"] == pytest.approx(1.0, abs=1e-6)
        assert result["mean_score"] == pytest.approx(3.0, abs=1e-6)

    def test_imperfect_predictions(self):
        predictions = np.array([1.0, 3.0, 5.0])
        labels = np.array([1.5, 2.5, 4.5])
        result = compute_metrics((predictions, labels))

        assert result["rmse"] > 0
        assert "R2" in result
        assert "pearson_corr" in result
        assert "pearson_pvalue" in result
        assert "st_dev_score" in result


class TestEvalComputeMetricsIdentical:
    def test_perfect_identical(self):
        predictions = np.array([100.0, 100.0, 100.0, 100.0])
        labels = np.array([100.0, 100.0, 100.0, 100.0])
        result = eval_compute_metrics_identical((predictions, labels))

        assert result["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert result["ratio_95"] == pytest.approx(100.0)
        assert result["ratio_99"] == pytest.approx(100.0)
        assert result["ratio_equals"] == pytest.approx(100.0)

    def test_partial_identical(self):
        predictions = np.array([100.0, 96.0, 90.0, 80.0])
        labels = np.array([100.0, 100.0, 100.0, 100.0])
        result = eval_compute_metrics_identical((predictions, labels))

        assert result["ratio_95"] == pytest.approx(50.0)
        assert result["ratio_99"] == pytest.approx(25.0)
        assert result["ratio_equals"] == pytest.approx(25.0)

    def test_low_predictions(self):
        predictions = np.array([50.0, 60.0, 70.0])
        labels = np.array([100.0, 100.0, 100.0])
        result = eval_compute_metrics_identical((predictions, labels))

        assert result["ratio_95"] == pytest.approx(0.0)
        assert result["ratio_99"] == pytest.approx(0.0)
        assert result["ratio_equals"] == pytest.approx(0.0)


class TestEvalComputeMetricsUnrelated:
    def test_perfect_unrelated(self):
        predictions = np.array([0.0, 0.0, 0.0, 0.0])
        labels = np.array([0.0, 0.0, 0.0, 0.0])
        result = eval_compute_metrics_unrelated((predictions, labels))

        assert result["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert result["ratio_1"] == pytest.approx(100.0)
        assert result["ratio_5"] == pytest.approx(100.0)
        assert result["ratio_equals"] == pytest.approx(100.0)

    def test_partial_unrelated(self):
        predictions = np.array([0.0, 0.4, 3.0, 10.0])
        labels = np.array([0.0, 0.0, 0.0, 0.0])
        result = eval_compute_metrics_unrelated((predictions, labels))

        assert result["ratio_equals"] == pytest.approx(50.0)
        assert result["ratio_5"] == pytest.approx(75.0)

    def test_high_predictions(self):
        predictions = np.array([50.0, 60.0, 70.0])
        labels = np.array([0.0, 0.0, 0.0])
        result = eval_compute_metrics_unrelated((predictions, labels))

        assert result["ratio_1"] == pytest.approx(0.0)
        assert result["ratio_5"] == pytest.approx(0.0)
        assert result["ratio_equals"] == pytest.approx(0.0)


class TestDivergenceFlagging:
    """C1: a run whose predictions contain NaN/Inf is flagged, never rewritten to zeros."""

    def test_clean_predictions_are_not_flagged(self):
        predictions = np.array([10.0, 50.0, 90.0])
        labels = np.array([12.0, 45.0, 95.0])
        result = compute_metrics((predictions, labels))

        assert result[DIVERGED_KEY] == 0.0
        assert result["rmse"] == pytest.approx(root_mean_squared_error(labels, predictions))
        assert result["mean_score"] == pytest.approx(50.0)

    def test_clean_predictions_keep_legitimate_zeros(self):
        # Zero is a legitimate label on the 0-100 scale: an honest zero-predictor stays healthy.
        predictions = np.array([0.0, 0.0, 0.0, 0.0])
        labels = np.array([0.0, 0.0, 0.0, 0.0])
        result = eval_compute_metrics_unrelated((predictions, labels))

        assert result[DIVERGED_KEY] == 0.0
        assert result["ratio_equals"] == pytest.approx(100.0)

    def test_nan_predictions_are_flagged_not_zeroed(self):
        predictions = np.array([np.nan, np.nan, np.nan, 40.0])
        labels = np.array([0.0, 50.0, 100.0, 40.0])
        result = compute_metrics((predictions, labels))

        assert result[DIVERGED_KEY] == 1.0
        assert result[N_NON_FINITE_KEY] == 3.0
        # The old behaviour reported rmse == root_mean_squared_error(labels, [0, 0, 0, 40]) ~= 64.55,
        # a plausible number for a zero-predictor. It must now be NaN instead.
        assert math.isnan(result["rmse"])
        assert math.isnan(result["mean_score"])
        assert math.isnan(result["st_dev_score"])
        assert math.isnan(result["pearson_corr"])

    def test_diverged_metrics_keep_the_same_keys_as_a_healthy_run(self):
        labels = np.array([0.0, 50.0, 100.0])
        healthy = compute_metrics((np.array([1.0, 51.0, 99.0]), labels))
        diverged = compute_metrics((np.array([np.nan, 51.0, 99.0]), labels))

        assert set(healthy) | {N_NON_FINITE_KEY} == set(diverged)

    def test_inf_predictions_are_flagged(self):
        predictions = np.array([np.inf, -np.inf, 50.0])
        labels = np.array([100.0, 0.0, 50.0])
        result = compute_metrics((predictions, labels))

        assert result[DIVERGED_KEY] == 1.0
        assert result[N_NON_FINITE_KEY] == 2.0

    def test_identical_holdout_is_flagged(self):
        predictions = np.array([np.nan, 100.0])
        labels = np.array([100.0, 100.0])
        result = eval_compute_metrics_identical((predictions, labels))

        assert result[DIVERGED_KEY] == 1.0
        # A diverged run must not be credited with a sanity-check ratio.
        assert math.isnan(result["ratio_95"])
        assert math.isnan(result["ratio_equals"])

    def test_unrelated_holdout_is_flagged(self):
        predictions = np.array([np.nan, 0.0])
        labels = np.array([0.0, 0.0])
        result = eval_compute_metrics_unrelated((predictions, labels))

        assert result[DIVERGED_KEY] == 1.0
        # This is the exact failure of the diagnostic: NaN -> 0.0 used to score 100% here.
        assert math.isnan(result["ratio_equals"])
        assert math.isnan(result["ratio_5"])


class TestSanitizePredictions:
    """C1: the helper counts the damage, it no longer repairs it."""

    def test_squeezes_column_vector(self):
        predictions, n_non_finite = _sanitize_predictions(np.array([[1.0], [2.0], [3.0]]))

        assert predictions.shape == (3,)
        assert n_non_finite == 0

    def test_counts_without_rewriting(self):
        predictions, n_non_finite = _sanitize_predictions(np.array([np.nan, 2.0, np.inf]))

        assert n_non_finite == 2
        assert math.isnan(predictions[0])
        assert predictions[1] == pytest.approx(2.0)
        assert math.isinf(predictions[2])
