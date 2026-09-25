"""Tests for the H1 calibration audit."""

import math

import numpy as np

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


# --- the clamped head, added because the score domain is closed -----------------------


def test_the_score_domain_is_closed_at_both_ends():
    """A null score exists and means no meaning preserved; 100 means fully preserved.
    100*sigmoid maps onto the OPEN interval (0, 100), so neither endpoint is reachable,
    while 31 to 48 percent of the v2 training labels sit exactly on one of them."""
    from training.calibration import percent_from_logits

    reachable = percent_from_logits(np.array([-50.0, 50.0]), "clamped")
    assert reachable[0] == 0.0
    assert reachable[1] == 100.0


def test_the_sigmoid_head_does_not_reach_the_endpoints_at_realistic_logits():
    """Not a claim of strict unreachability: in float64 the sigmoid saturates to exactly
    1.0 somewhere past a logit of 37, so it does numerically return 100. The point is that
    training never gets there. The measured identical predictions sit at 96.36, which is a
    logit of 3.3, and the MSE gradient through the sigmoid at that point is already about
    0.035 per unit of error."""
    from training.calibration import percent_from_logits

    reached = percent_from_logits(np.array([-3.3, 3.3]), "sigmoid")
    assert reached[0] > 0.0
    assert reached[1] < 100.0
    assert reached[1] == pytest.approx(96.4, abs=0.2)


def test_the_gradient_argument_the_clamped_head_answers():
    """d/dz of 100*sigmoid(z) collapses as z grows, so the optimiser stalls short of the
    endpoint. The clamped head has a gradient of 1 right up to the boundary."""
    for logit, ceiling in ((3.3, 3.5), (6.9, 0.11), (10.0, 0.005)):
        slope = 100.0 * (1.0 / (1.0 + np.exp(-logit))) * (1.0 - 1.0 / (1.0 + np.exp(-logit)))
        assert slope < ceiling


def test_reaching_ninety_nine_with_a_sigmoid_needs_a_large_logit():
    """Which is why the measured identical predictions pile up at 96.36: the MSE gradient
    has all but vanished by the time the logit gets there."""
    from training.calibration import percent_from_logits

    assert percent_from_logits(np.array([4.6]), "sigmoid")[0] == pytest.approx(99.0, abs=0.1)


def test_the_clamped_head_maps_the_unit_targets_like_the_other_bounded_heads():
    from training.calibration import targets_for_head

    assert list(targets_for_head(np.array([0.0, 50.0, 100.0]), "clamped")) == [0.0, 0.5, 1.0]


def test_the_clamped_unit_output_is_bounded_on_both_sides():
    from training.calibration import unit_from_logits

    values = unit_from_logits(np.array([-3.0, 0.5, 4.0]), "clamped")
    assert values.min() >= 0.0
    assert values.max() <= 1.0


def test_an_unknown_head_is_refused_rather_than_guessed():
    from training.calibration import percent_from_logits

    with pytest.raises(ValueError, match="Unknown output head"):
        percent_from_logits(np.array([0.0]), "softmax")
