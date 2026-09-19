"""Tests for the H1 calibration audit."""

import math

import pytest

from diagnostics.calibration_audit import (
    COLLAPSE_STD_THRESHOLD,
    audit_run,
    audit_runs,
    optimal_affine_rmse,
)

LABEL_STD = 37.01


def _summary(**overrides) -> dict:
    summary = {
        "test/rmse": 41.69,
        "test/pearson_corr": 0.795,
        "test/mean_score": 31.31,
        "test/st_dev_score": 14.49,
    }
    summary.update(overrides)
    return summary


def _run(name="run", checkpoint="ckpt", augmentation="swap", **overrides):
    return audit_run(name, checkpoint, augmentation, _summary(**overrides), LABEL_STD)


# --- optimal_affine_rmse -------------------------------------------------------------


def test_perfect_correlation_leaves_no_recoverable_error():
    assert optimal_affine_rmse(LABEL_STD, 1.0) == pytest.approx(0.0)


def test_zero_correlation_cannot_beat_predicting_the_label_mean():
    assert optimal_affine_rmse(LABEL_STD, 0.0) == pytest.approx(LABEL_STD)


def test_a_perfectly_inverted_ranking_is_still_fully_rescalable():
    """An affine map may have a negative slope, so r = -1 is as recoverable as r = +1."""
    assert optimal_affine_rmse(LABEL_STD, -1.0) == pytest.approx(0.0)


def test_matches_the_closed_form_on_the_observed_sweep_value():
    assert optimal_affine_rmse(37.01, 0.795) == pytest.approx(22.45, abs=0.01)


def test_the_documented_v2_target_needs_the_documented_correlation():
    """PRODUIT.md targets RMSE < 15, which the closed form puts at Pearson 0.914."""
    assert optimal_affine_rmse(37.01, 0.914) == pytest.approx(15.0, abs=0.05)


def test_a_nan_correlation_falls_back_to_the_label_spread():
    assert optimal_affine_rmse(LABEL_STD, float("nan")) == pytest.approx(LABEL_STD)


def test_a_correlation_outside_minus_one_to_one_is_clamped_not_crashed():
    """Floating point can hand back 1.0000000002; that must not produce a NaN sqrt."""
    assert optimal_affine_rmse(LABEL_STD, 1.0000000002) == pytest.approx(0.0)


# --- audit_run -----------------------------------------------------------------------


def test_a_healthy_run_is_not_flagged_as_collapsed():
    audit = _run()
    assert audit is not None
    assert not audit.collapsed


def test_a_run_without_test_metrics_is_skipped():
    assert audit_run("r", "c", "swap", {"eval/rmse": 10.0}, LABEL_STD) is None


def test_a_constant_output_run_is_flagged_as_collapsed():
    audit = _run(**{"test/st_dev_score": 0.0})
    assert audit.collapsed


def test_a_run_just_under_the_collapse_threshold_is_flagged():
    audit = _run(**{"test/st_dev_score": COLLAPSE_STD_THRESHOLD / 2})
    assert audit.collapsed


def test_an_undefined_pearson_is_treated_as_collapse():
    """Pearson is undefined exactly when the predictions have no variance."""
    audit = _run(**{"test/pearson_corr": float("nan")})
    assert audit.collapsed


def test_recoverable_fraction_quantifies_the_calibration_loss():
    audit = _run()
    assert audit.recoverable_fraction == pytest.approx(1 - 22.45 / 41.69, abs=0.005)


def test_a_run_already_at_the_affine_floor_has_nothing_left_to_recover():
    audit = _run(**{"test/rmse": optimal_affine_rmse(LABEL_STD, 0.795)})
    assert audit.recoverable_fraction == pytest.approx(0.0, abs=1e-9)


def test_string_valued_summary_fields_do_not_crash_the_audit():
    """wandb hands back strings for some fields; the audit must degrade, not raise."""
    audit = _run(**{"test/st_dev_score": "NaN"})
    assert audit is not None
    assert audit.collapsed


# --- audit_runs and aggregation ------------------------------------------------------


def _record(name, checkpoint, **overrides):
    return (name, checkpoint, "swap", _summary(**overrides))


def test_audit_runs_counts_collapsed_runs():
    report = audit_runs(
        [
            _record("a", "big"),
            _record("b", "big", **{"test/st_dev_score": 0.0}),
            _record("c", "small"),
        ],
        62.66,
        LABEL_STD,
    )
    assert report.n_runs == 3
    assert report.n_collapsed == 1


def test_by_checkpoint_excludes_collapsed_runs_from_the_medians():
    """A dead run reported as a zero-predictor must not drag the aggregate."""
    report = audit_runs(
        [
            _record("a", "big", **{"test/rmse": 40.0}),
            _record("b", "big", **{"test/rmse": 73.3, "test/st_dev_score": 0.0}),
        ],
        62.66,
        LABEL_STD,
    )
    stats = report.by_checkpoint()
    assert stats["big"]["n"] == 1
    assert stats["big"]["rmse"] == pytest.approx(40.0)


def test_a_checkpoint_whose_runs_all_collapsed_disappears_from_the_aggregate():
    report = audit_runs([_record("a", "dead", **{"test/st_dev_score": 0.0})], 62.66, LABEL_STD)
    assert "dead" not in report.by_checkpoint()


def test_runs_missing_test_metrics_are_dropped_before_aggregation():
    report = audit_runs([("a", "big", "swap", {}), _record("b", "big")], 62.66, LABEL_STD)
    assert report.n_runs == 1


def test_the_audit_reproduces_the_observed_amplitude_compression():
    """The sweep's own numbers must come out as compression, not as a modelling failure."""
    report = audit_runs([_record("a", "deberta-v2-xlarge")], 62.66, LABEL_STD)
    audit = report.runs[0]
    assert audit.pred_std < LABEL_STD / 2
    assert audit.pearson > 0.75
    assert audit.recoverable_fraction > 0.4
    assert not math.isnan(audit.rmse_after_affine)
