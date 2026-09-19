"""Tests for the bounded output heads (C3) and the recalibration module (C4)."""

import math
import warnings

import numpy as np
import pytest

from diagnostics.calibration_audit import DEFAULT_LABEL_STD, optimal_affine_rmse
from training.calibration import (
    MIN_ISOTONIC_DEV_SIZE,
    SCORE_MAX,
    SCORE_MIN,
    AffineCalibrator,
    CalibrationError,
    CalibrationLeakError,
    IsotonicOverfitWarning,
    calibrate_dev_to_test,
    calibration_gain,
    clip_to_score_range,
    fit_calibrator,
    percent_from_logits,
    rmse,
    sigmoid_percent,
    synthetic_predictions,
    targets_for_head,
    unit_from_logits,
    unit_percent,
)

EXTREME_LOGITS = np.array([-1e30, -1e6, -800.0, -50.0, -1.0, 0.0, 1.0, 50.0, 800.0, 1e6, 1e30])


class TestBoundedHeads:
    """C3: whatever the head, what gets reported lives on the 0-100 scale."""

    def test_sigmoid_head_stays_in_range_for_extreme_logits(self):
        scores = percent_from_logits(EXTREME_LOGITS, head="sigmoid")

        assert np.all(np.isfinite(scores))
        assert np.all(scores >= SCORE_MIN)
        assert np.all(scores <= SCORE_MAX)

    def test_sigmoid_head_saturates_on_both_sides(self):
        scores = sigmoid_percent(np.array([-1e30, 0.0, 1e30]))

        assert scores[0] == pytest.approx(0.0, abs=1e-9)
        assert scores[1] == pytest.approx(50.0)
        assert scores[2] == pytest.approx(100.0, abs=1e-9)

    def test_sigmoid_head_is_monotone(self):
        scores = sigmoid_percent(np.sort(EXTREME_LOGITS))

        assert np.all(np.diff(scores) >= 0)

    def test_normalized_head_stays_in_range_for_extreme_logits(self):
        # The normalized head stays linear, so the clip is what bounds it.
        scores = percent_from_logits(np.array([-1e6, -0.5, 0.0, 0.42, 1.0, 1e6]), head="normalized")

        assert np.all(scores >= SCORE_MIN)
        assert np.all(scores <= SCORE_MAX)

    def test_normalized_head_rescales_the_unit_interval(self):
        scores = unit_percent(np.array([0.0, 0.25, 0.5, 1.0]))

        assert scores.tolist() == [0.0, 25.0, 50.0, 100.0]

    def test_linear_head_is_left_untouched(self):
        # The v1 head is unbounded, and it must stay unbounded: changing it silently would
        # make every number of the sweep incomparable.
        logits = np.array([-40.0, 0.0, 250.0])

        assert percent_from_logits(logits, head="linear").tolist() == logits.tolist()

    def test_targets_are_put_on_the_scale_the_loss_expects(self):
        labels = np.array([0.0, 50.0, 100.0])

        assert targets_for_head(labels, head="linear").tolist() == [0.0, 50.0, 100.0]
        assert targets_for_head(labels, head="sigmoid").tolist() == [0.0, 0.5, 1.0]
        assert targets_for_head(labels, head="normalized").tolist() == [0.0, 0.5, 1.0]

    def test_unit_and_percent_views_agree(self):
        logits = np.array([-3.0, 0.0, 2.5])

        for head in ("sigmoid", "normalized"):
            unit = unit_from_logits(logits, head=head)
            percent = percent_from_logits(logits, head=head)
            # The normalized head clips at the percent scale only, so compare where it does not bite.
            inside = (percent > SCORE_MIN) & (percent < SCORE_MAX)
            assert np.allclose(np.asarray(unit)[inside] * SCORE_MAX, np.asarray(percent)[inside])

    def test_linear_head_unit_view_divides_by_one_hundred(self):
        assert unit_from_logits(np.array([0.0, 50.0, 100.0]), head="linear").tolist() == [0.0, 0.5, 1.0]

    def test_unknown_head_is_refused(self):
        with pytest.raises(ValueError, match="Unknown output head"):
            percent_from_logits(np.array([0.0]), head="softmax")
        with pytest.raises(ValueError, match="Unknown output head"):
            targets_for_head(np.array([0.0]), head="softmax")
        with pytest.raises(ValueError, match="Unknown output head"):
            unit_from_logits(np.array([0.0]), head="softmax")

    def test_clip_to_score_range(self):
        assert clip_to_score_range(np.array([-10.0, 42.0, 140.0])).tolist() == [0.0, 42.0, 100.0]


class TestSyntheticSample:
    def test_sample_correlation_is_exactly_the_requested_one(self):
        predictions, labels = synthetic_predictions(500, 0.802, seed=7)

        assert float(np.corrcoef(predictions, labels)[0, 1]) == pytest.approx(0.802, abs=1e-12)

    def test_moments_are_the_requested_ones(self):
        predictions, labels = synthetic_predictions(500, 0.8, label_mean=62.66, label_std=37.01, seed=3)

        assert float(np.mean(labels)) == pytest.approx(62.66)
        assert float(np.std(labels)) == pytest.approx(37.01)
        assert float(np.std(predictions)) == pytest.approx(14.49)

    def test_refuses_impossible_arguments(self):
        with pytest.raises(ValueError):
            synthetic_predictions(2, 0.5)
        with pytest.raises(ValueError):
            synthetic_predictions(100, 1.5)


class TestAffineFloor:
    """C4: the affine fit reaches ``sigma_y * sqrt(1 - r^2)``, and nothing beats it."""

    @pytest.mark.parametrize("pearson", [0.784, 0.795, 0.802, 0.914])
    def test_affine_fit_reaches_the_theoretical_floor(self, pearson):
        # Moments chosen so the recalibrated values stay inside [0, 100] and the clip is a
        # no-op: the claim under test is the fit, not the clip.
        predictions, labels = synthetic_predictions(
            800, pearson, label_mean=50.0, label_std=10.0, pred_mean=20.0, pred_std=4.0, seed=11
        )
        calibrator = fit_calibrator(predictions, labels, split="dev", method="affine")
        calibrated = calibrator.transform(predictions)

        floor = optimal_affine_rmse(float(np.std(labels)), pearson)
        assert rmse(calibrated, labels) == pytest.approx(floor, rel=1e-9)

    def test_no_other_affine_rescaling_beats_the_fitted_one(self):
        predictions, labels = synthetic_predictions(
            400, 0.8, label_mean=50.0, label_std=10.0, pred_mean=20.0, pred_std=4.0, seed=5
        )
        calibrator = fit_calibrator(predictions, labels, split="dev")
        best = rmse(calibrator.transform(predictions), labels)

        for slope_delta, intercept_delta in [(0.1, 0.0), (-0.1, 0.0), (0.0, 2.0), (0.0, -2.0)]:
            worse = AffineCalibrator(
                slope=calibrator.slope + slope_delta,
                intercept=calibrator.intercept + intercept_delta,
                n_fit=calibrator.n_fit,
                fit_split="dev",
            )
            assert rmse(worse.transform(predictions), labels) > best

    def test_the_fit_undoes_the_compression_of_the_sweep(self):
        predictions, labels = synthetic_predictions(600, 0.802, seed=13)
        calibrator = fit_calibrator(predictions, labels, split="dev")
        calibrated = calibrator.transform(predictions)
        gain = calibration_gain(predictions, labels, calibrated)

        # The diagnostic measures 40 to 55 % of the RMSE as recoverable on the real sweep.
        assert 0.30 < gain["recovered_fraction"] < 0.70
        assert gain["rmse_after"] < gain["rmse_before"]
        # The compression is undone: the predicted amplitude and position both move
        # towards the labels. The clip to [0, 100] keeps the gain slightly under the
        # raw factor of 2.06 the fitted slope applies.
        label_std, label_mean = float(np.std(labels)), float(np.mean(labels))
        assert abs(float(np.std(calibrated)) - label_std) < abs(float(np.std(predictions)) - label_std)
        assert abs(float(np.mean(calibrated)) - label_mean) < abs(float(np.mean(predictions)) - label_mean)
        assert float(np.std(calibrated)) > 1.8 * float(np.std(predictions))

    def test_recalibrated_scores_stay_inside_the_scale(self):
        predictions, labels = synthetic_predictions(600, 0.6, label_std=DEFAULT_LABEL_STD, seed=17)
        calibrator = fit_calibrator(predictions, labels, split="dev")
        calibrated = calibrator.transform(np.concatenate([predictions, np.array([-500.0, 500.0])]))

        assert np.all(calibrated >= SCORE_MIN)
        assert np.all(calibrated <= SCORE_MAX)


class TestLeakGuard:
    """C4: fitting on the split you then report is the mistake the API must make hard."""

    def test_fitting_on_the_test_split_raises(self):
        predictions, labels = synthetic_predictions(200, 0.8, seed=1)

        with pytest.raises(CalibrationLeakError, match="leak"):
            fit_calibrator(predictions, labels, split="test")

    @pytest.mark.parametrize("split", ["test", "TEST", " Test ", "holdout", "holdout_identical"])
    def test_every_evaluation_split_is_refused(self, split):
        predictions, labels = synthetic_predictions(200, 0.8, seed=1)

        with pytest.raises(CalibrationLeakError):
            fit_calibrator(predictions, labels, split=split)

    def test_the_escape_hatch_warns_instead_of_raising(self):
        predictions, labels = synthetic_predictions(200, 0.8, seed=1)

        with pytest.warns(UserWarning, match="leaked"):
            calibrator = fit_calibrator(predictions, labels, split="test", allow_evaluation_split=True)

        assert calibrator.fit_split == "test"

    def test_the_dev_split_is_allowed(self):
        predictions, labels = synthetic_predictions(200, 0.8, seed=1)
        calibrator = fit_calibrator(predictions, labels, split="dev")

        assert calibrator.fit_split == "dev"
        assert calibrator.n_fit == 200

    def test_calibrate_dev_to_test_fits_on_dev_only(self):
        dev_predictions, dev_labels = synthetic_predictions(200, 0.8, seed=2)
        test_predictions, test_labels = synthetic_predictions(200, 0.8, seed=3)

        calibrated, calibrator = calibrate_dev_to_test(dev_predictions, dev_labels, test_predictions)

        assert calibrator.fit_split == "dev"
        assert calibrator.n_fit == 200
        assert rmse(calibrated, test_labels) < rmse(test_predictions, test_labels)


class TestIdempotence:
    """C4: recalibrating an already calibrated set of predictions changes nothing."""

    def test_refitting_on_calibrated_predictions_gives_the_identity(self):
        predictions, labels = synthetic_predictions(
            500, 0.8, label_mean=50.0, label_std=10.0, pred_mean=20.0, pred_std=4.0, seed=23
        )
        first = fit_calibrator(predictions, labels, split="dev")
        once = first.transform(predictions)

        second = fit_calibrator(once, labels, split="dev")

        assert second.slope == pytest.approx(1.0, abs=1e-9)
        assert second.intercept == pytest.approx(0.0, abs=1e-8)
        assert np.allclose(second.transform(once), once)

    def test_isotonic_calibration_is_idempotent(self):
        predictions, labels = synthetic_predictions(
            MIN_ISOTONIC_DEV_SIZE * 2, 0.8, label_mean=50.0, label_std=10.0, pred_mean=20.0, pred_std=4.0, seed=29
        )
        first = fit_calibrator(predictions, labels, split="dev", method="isotonic")
        once = first.transform(predictions)
        second = fit_calibrator(once, labels, split="dev", method="isotonic")

        assert np.allclose(second.transform(once), once, atol=1e-9)


class TestCalibratorDescription:
    """The audit trail: a calibrator says what it is and what it was fitted on."""

    def test_affine_calibrator_describes_its_fit(self):
        predictions, labels = synthetic_predictions(120, 0.8, seed=59)
        description = fit_calibrator(predictions, labels, split="dev").describe()

        assert description.startswith("affine(slope=")
        assert "120 rows of 'dev'" in description

    def test_isotonic_calibrator_describes_its_fit(self):
        predictions, labels = synthetic_predictions(MIN_ISOTONIC_DEV_SIZE + 5, 0.8, seed=61)
        description = fit_calibrator(predictions, labels, split="dev", method="isotonic").describe()

        assert description == f"isotonic fitted on {MIN_ISOTONIC_DEV_SIZE + 5} rows of 'dev'"


class TestIsotonicGuardRail:
    def test_a_small_dev_split_warns(self):
        # The CSMD dev split holds 95 rows.
        predictions, labels = synthetic_predictions(95, 0.8, seed=31)

        with pytest.warns(IsotonicOverfitWarning, match="generalize"):
            fit_calibrator(predictions, labels, split="dev", method="isotonic")

    def test_a_large_dev_split_does_not_warn(self):
        predictions, labels = synthetic_predictions(MIN_ISOTONIC_DEV_SIZE + 50, 0.8, seed=37)

        with warnings.catch_warnings():
            warnings.simplefilter("error", IsotonicOverfitWarning)
            fit_calibrator(predictions, labels, split="dev", method="isotonic")

    def test_isotonic_output_stays_inside_the_scale(self):
        predictions, labels = synthetic_predictions(MIN_ISOTONIC_DEV_SIZE + 10, 0.7, seed=41)
        calibrator = fit_calibrator(predictions, labels, split="dev", method="isotonic")
        calibrated = calibrator.transform(np.array([-1e6, 0.0, 30.0, 1e6]))

        assert np.all(calibrated >= SCORE_MIN)
        assert np.all(calibrated <= SCORE_MAX)

    def test_isotonic_is_monotone(self):
        predictions, labels = synthetic_predictions(MIN_ISOTONIC_DEV_SIZE + 10, 0.7, seed=43)
        calibrator = fit_calibrator(predictions, labels, split="dev", method="isotonic")
        calibrated = calibrator.transform(np.sort(predictions))

        assert np.all(np.diff(calibrated) >= -1e-9)


class TestUnusableInputs:
    def test_constant_predictions_are_refused(self):
        with pytest.raises(CalibrationError, match="collapsed run"):
            fit_calibrator(np.full(50, 37.0), np.linspace(0, 100, 50), split="dev")

    def test_non_finite_predictions_are_refused(self):
        predictions = np.linspace(0, 100, 50)
        predictions[3] = np.nan

        with pytest.raises(CalibrationError, match="non-finite"):
            fit_calibrator(predictions, np.linspace(0, 100, 50), split="dev")

    def test_length_mismatch_is_refused(self):
        with pytest.raises(CalibrationError, match="same length"):
            fit_calibrator(np.linspace(0, 100, 50), np.linspace(0, 100, 49), split="dev")

    def test_empty_input_is_refused(self):
        with pytest.raises(CalibrationError, match="empty"):
            fit_calibrator(np.array([]), np.array([]), split="dev")

    def test_unknown_method_is_refused(self):
        predictions, labels = synthetic_predictions(50, 0.8, seed=47)

        with pytest.raises(ValueError, match="Unknown calibration method"):
            fit_calibrator(predictions, labels, split="dev", method="quantile")


class TestCalibrationGain:
    def test_gain_reports_the_floor_and_the_recovered_share(self):
        predictions, labels = synthetic_predictions(400, 0.8, seed=53)
        calibrated, _ = calibrate_dev_to_test(predictions, labels, predictions)
        gain = calibration_gain(predictions, labels, calibrated)

        assert gain["pearson"] == pytest.approx(0.8, abs=1e-9)
        assert gain["rmse_affine_floor"] == pytest.approx(optimal_affine_rmse(float(np.std(labels)), 0.8), rel=1e-9)
        assert 0.0 < gain["recovered_fraction"] < 1.0

    def test_gain_of_an_already_perfect_prediction_is_zero(self):
        labels = np.linspace(0, 100, 50)
        gain = calibration_gain(labels, labels, labels)

        assert gain["rmse_before"] == pytest.approx(0.0)
        assert gain["recovered_fraction"] == pytest.approx(0.0)
        assert math.isfinite(gain["rmse_affine_floor"])
