"""Tests for the collapse detection callback (C2)."""

import pytest

from diagnostics.calibration_audit import COLLAPSE_STD_THRESHOLD
from training.callbacks import (
    DEFAULT_COLLAPSE_PATIENCE,
    CollapseDetector,
    PredictionCollapseCallback,
)


class _Control:
    """Stand-in for ``TrainerControl``: only the stop flag matters here."""

    def __init__(self) -> None:
        self.should_training_stop = False


class _State:
    """Stand-in for ``TrainerState``."""

    def __init__(self, epoch: float = 1.0) -> None:
        self.epoch = epoch


class TestCollapseDetector:
    def test_rejects_a_patience_below_one(self):
        with pytest.raises(ValueError):
            CollapseDetector(patience=0)

    def test_rejects_a_negative_threshold(self):
        with pytest.raises(ValueError):
            CollapseDetector(std_threshold=-1.0)

    def test_does_not_trigger_at_patience_minus_one(self):
        detector = CollapseDetector(std_threshold=1.0, patience=3)

        assert detector.update(0.0) is False
        assert detector.update(0.0) is False
        assert detector.consecutive_degenerate == 2
        assert detector.collapsed is False

    def test_triggers_at_patience(self):
        detector = CollapseDetector(std_threshold=1.0, patience=3)

        detector.update(0.0)
        detector.update(0.0)

        assert detector.update(0.0) is True
        assert detector.collapsed is True

    def test_does_not_trigger_when_the_variance_comes_back(self):
        detector = CollapseDetector(std_threshold=1.0, patience=3)

        detector.update(0.0)
        detector.update(0.0)
        detector.update(14.5)  # the spread recovers, the count restarts from zero
        detector.update(0.0)
        detector.update(0.0)

        assert detector.collapsed is False
        assert detector.consecutive_degenerate == 2

    def test_a_healthy_run_never_triggers(self):
        detector = CollapseDetector(patience=2)

        for spread in [16.88, 14.49, 9.22, 10.00, 37.01]:
            assert detector.update(spread) is False

    def test_a_non_finite_spread_counts_as_degenerate(self):
        # A NaN spread is not "unknown", it is what a dead run reports.
        detector = CollapseDetector(patience=2)

        assert detector.update(float("nan")) is False
        assert detector.update(float("nan")) is True

    def test_a_missing_metric_leaves_the_verdict_untouched(self):
        detector = CollapseDetector(std_threshold=1.0, patience=2)

        detector.update(0.0)
        assert detector.update(None) is False
        assert detector.consecutive_degenerate == 1

    def test_default_threshold_is_the_audit_threshold(self):
        detector = CollapseDetector()

        assert detector.std_threshold == COLLAPSE_STD_THRESHOLD
        assert detector.patience == DEFAULT_COLLAPSE_PATIENCE
        assert detector.is_degenerate(COLLAPSE_STD_THRESHOLD / 2)
        assert not detector.is_degenerate(COLLAPSE_STD_THRESHOLD * 2)

    def test_reset_forgets_the_observations(self):
        detector = CollapseDetector(std_threshold=1.0, patience=2)

        detector.update(0.0)
        detector.reset()

        assert detector.update(0.0) is False


class TestPredictionCollapseCallback:
    def test_does_not_stop_a_healthy_run(self):
        callback = PredictionCollapseCallback(std_threshold=1.0, patience=2)
        control = _Control()

        for spread in [16.88, 0.0, 14.49, 0.0]:
            callback.on_evaluate(None, _State(), control, metrics={"eval_st_dev_score": spread, "eval_diverged": 0.0})

        assert control.should_training_stop is False
        assert callback.stop_reason is None

    def test_stops_after_patience_degenerate_evaluations(self):
        callback = PredictionCollapseCallback(std_threshold=1.0, patience=2)
        control = _Control()

        callback.on_evaluate(None, _State(1.0), control, metrics={"eval_st_dev_score": 0.11})
        assert control.should_training_stop is False

        callback.on_evaluate(None, _State(2.0), control, metrics={"eval_st_dev_score": 0.0})
        assert control.should_training_stop is True
        assert "standard deviation" in callback.stop_reason

    def test_stops_immediately_on_the_c1_divergence_flag(self):
        callback = PredictionCollapseCallback(patience=10)
        control = _Control()

        callback.on_evaluate(None, _State(44.0), control, metrics={"eval_diverged": 1.0, "eval_st_dev_score": "NaN"})

        assert control.should_training_stop is True
        assert "NaN" in callback.stop_reason

    def test_divergence_stop_can_be_disabled(self):
        callback = PredictionCollapseCallback(patience=10, stop_on_diverged=False)
        control = _Control()

        callback.on_evaluate(None, _State(), control, metrics={"eval_diverged": 1.0, "eval_st_dev_score": 14.0})

        assert control.should_training_stop is False

    def test_reads_the_metric_whatever_the_evaluation_prefix(self):
        callback = PredictionCollapseCallback(std_threshold=1.0, patience=1)
        control = _Control()

        callback.on_evaluate(None, _State(), control, metrics={"test_st_dev_score": 0.0})

        assert control.should_training_stop is True

    def test_an_evaluation_without_metrics_is_ignored(self):
        callback = PredictionCollapseCallback(std_threshold=1.0, patience=1)
        control = _Control()

        callback.on_evaluate(None, _State(), control, metrics=None)
        callback.on_evaluate(None, _State(), control, metrics={})

        assert control.should_training_stop is False

    def test_reproduces_the_diagnostic_trajectory(self):
        # deberta-v3-large / swap: constant from epoch 2, NaN at epoch 44, ran to epoch 58.
        callback = PredictionCollapseCallback(std_threshold=1.0, patience=3)
        control = _Control()
        trajectory = [0.11, 0.0, 0.0, 0.0, 0.0]

        stopped_at = None
        for epoch, spread in enumerate(trajectory, start=1):
            callback.on_evaluate(None, _State(float(epoch)), control, metrics={"eval_st_dev_score": spread})
            if control.should_training_stop and stopped_at is None:
                stopped_at = epoch

        # Epoch 1 already sits at 0.11, degenerate under a threshold of 1.0: three
        # consecutive degenerate evaluations are reached at epoch 3 instead of epoch 58.
        assert stopped_at == 3
