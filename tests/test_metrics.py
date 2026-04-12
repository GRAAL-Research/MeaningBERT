import numpy as np
import pytest

from src.training.metrics.metrics import compute_metrics, eval_compute_metrics_identical, eval_compute_metrics_unrelated


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
