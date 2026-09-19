"""Tests for the table helpers: C1 exclusion of dead runs, C5 compression reporting."""

import math

import pytest

from diagnostics.calibration_audit import COLLAPSE_STD_THRESHOLD, DEFAULT_LABEL_MEAN, DEFAULT_LABEL_STD
from figures_generator.compression_report import (
    compression_ratio,
    drop_unusable_runs,
    format_mean_std,
    is_flagged_diverged,
    is_usable_run,
    label_reference_caption,
    latex_mean_std,
    label_reference_line,
    looks_collapsed,
    pred_moment_columns,
)


def _healthy(**overrides) -> dict:
    """A summary with the shape of a healthy run of the sweep."""
    summary = {
        "test/rmse": 41.69,
        "test/pearson_corr": 0.795,
        "test/mean_score": 31.31,
        "test/st_dev_score": 14.49,
        "test/diverged": 0.0,
    }
    summary.update(overrides)
    return summary


class TestDivergenceFlag:
    def test_flag_set_is_detected(self):
        assert is_flagged_diverged(_healthy(**{"test/diverged": 1.0}))

    def test_flag_cleared_is_not_detected(self):
        assert not is_flagged_diverged(_healthy())

    def test_flag_detected_whatever_the_prefix(self):
        assert is_flagged_diverged({"train/test/identical_sentences_diverged": 1.0})
        assert is_flagged_diverged({"eval_diverged": 1.0})

    def test_absent_flag_is_not_a_divergence(self):
        # Runs of the previous sweep predate the flag entirely.
        assert not is_flagged_diverged({"test/rmse": 37.1})

    def test_a_key_merely_containing_diverged_is_ignored(self):
        assert not is_flagged_diverged({"test/diverged_count_note": 1.0})


class TestCollapseSignature:
    def test_constant_output_is_collapsed(self):
        assert looks_collapsed(_healthy(**{"test/st_dev_score": 0.0}))

    def test_spread_just_under_the_threshold_is_collapsed(self):
        assert looks_collapsed(_healthy(**{"test/st_dev_score": COLLAPSE_STD_THRESHOLD / 2}))

    def test_spread_just_over_the_threshold_is_not(self):
        assert not looks_collapsed(_healthy(**{"test/st_dev_score": COLLAPSE_STD_THRESHOLD * 2}))

    def test_undefined_pearson_is_collapsed(self):
        assert looks_collapsed(_healthy(**{"test/pearson_corr": "NaN"}))

    def test_string_nan_spread_is_collapsed(self):
        # wandb hands NaN back as a string for some runs; that is exactly a dead run.
        assert looks_collapsed(_healthy(**{"test/st_dev_score": "NaN"}))

    def test_a_summary_without_those_metrics_is_not_judged(self):
        # Benchmark metric rows (BLEU, SARI...) carry no prediction moments.
        assert not looks_collapsed({"test/BLEU_rmse": 55.0, "test/BLEU_pearson_corr": 0.3})

    def test_healthy_run_is_usable(self):
        assert is_usable_run(_healthy())

    def test_the_diagnostic_dead_run_is_not_usable(self):
        # The run of the diagnostic: constant output, NaN Pearson, RMSE 73.30.
        dead = _healthy(**{"test/rmse": 73.30, "test/st_dev_score": 0.0, "test/pearson_corr": "NaN"})
        assert not is_usable_run(dead)


class TestDropUnusableRuns:
    def test_drops_only_the_unusable(self):
        runs = [_healthy(), _healthy(**{"test/diverged": 1.0}), _healthy(**{"test/st_dev_score": 0.0})]
        kept = drop_unusable_runs(runs, "group")

        assert len(kept) == 1
        assert kept[0]["test/rmse"] == pytest.approx(41.69)

    def test_consumes_a_generator_once(self):
        kept = drop_unusable_runs(iter([_healthy(), _healthy(**{"test/diverged": 1.0})]))

        assert len(kept) == 1

    def test_an_all_healthy_group_is_untouched(self):
        runs = [_healthy(), _healthy(**{"test/rmse": 37.1})]

        assert drop_unusable_runs(runs) == runs


class TestCompressionReporting:
    def test_compression_ratio_matches_the_diagnostic(self):
        # deberta-v2-xlarge: pred_std 14.49 against a label std of 37.01.
        assert compression_ratio(14.49) == pytest.approx(DEFAULT_LABEL_STD / 14.49)
        assert 2.0 < compression_ratio(14.49) < 4.0

    def test_compression_ratio_of_a_constant_output_is_not_a_number(self):
        assert math.isnan(compression_ratio(0.0))
        assert math.isnan(compression_ratio(float("nan")))

    def test_format_mean_std(self):
        assert format_mean_std([10.0, 20.0, 30.0]) == "20.00 +/- 10.00"
        assert format_mean_std([42.0]) == "42.00"
        assert format_mean_std([]) == "n/a"

    def test_format_mean_std_ignores_non_finite_values(self):
        assert format_mean_std([10.0, float("nan"), 30.0]) == "20.00 +/- 14.14"

    def test_latex_mean_std(self):
        assert latex_mean_std([10.0, 20.0, 30.0]) == "20.00" + r"$\pm$" + "10.00"
        assert latex_mean_std([42.0]) == "42.00"
        assert latex_mean_std([]) == "n/a"
        assert latex_mean_std([None, "x"]) == "n/a"

    def test_a_non_numeric_summary_value_is_not_trusted(self):
        # A summary value that is not a number cannot vouch for a live run.
        assert looks_collapsed(_healthy(**{"test/st_dev_score": "unavailable"}))

    def test_label_reference_carries_both_moments(self):
        line = label_reference_line()

        assert f"{DEFAULT_LABEL_MEAN:.2f}" in line
        assert f"{DEFAULT_LABEL_STD:.2f}" in line

    def test_label_reference_caption_carries_both_moments(self):
        caption = label_reference_caption()

        assert f"{DEFAULT_LABEL_MEAN:.2f}" in caption
        assert f"{DEFAULT_LABEL_STD:.2f}" in caption

    def test_pred_moment_columns_averages_the_group(self):
        moments = pred_moment_columns([_healthy(), _healthy(**{"test/mean_score": 41.31, "test/st_dev_score": 16.49})])

        assert moments["pred_mean"] == pytest.approx(36.31)
        assert moments["pred_std"] == pytest.approx(15.49)
        assert moments["compression"] == pytest.approx(DEFAULT_LABEL_STD / 15.49)

    def test_pred_moment_columns_on_an_empty_group(self):
        moments = pred_moment_columns([])

        assert moments["pred_mean"] is None
        assert moments["pred_std"] is None
