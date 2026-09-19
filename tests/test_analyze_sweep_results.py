"""Tests for the sweep summary tables: C1 exclusion and C5 compression columns."""

import csv

import pytest

from diagnostics.calibration_audit import DEFAULT_LABEL_MEAN, DEFAULT_LABEL_STD
from figures_generator.analyze_sweep_results import (
    compute_summary_table,
    filter_usable_runs,
    generate_latex_table,
    group_runs,
    save_csv,
)


def _run(name: str, checkpoint: str = "microsoft/deberta-v3-base", fold: int = 0, **overrides) -> dict:
    """A structured run as :func:`fetch_runs` builds it, with healthy sweep values."""
    run = {
        "name": name,
        "id": name,
        "created_at": "2026-04-01",
        "checkpoint": checkpoint,
        "augmentation": "swap",
        "fold": fold,
        "seed": 42,
        "summary": {"test/st_dev_score": 9.22, "test/pearson_corr": 0.784, "test/diverged": 0.0},
        "test/rmse": 51.41,
        "test/R2": -0.9,
        "test/pearson_corr": 0.784,
        "test/mean_score": 21.64,
        "test/st_dev_score": 9.22,
        "train/test/identical_sentences_ratio_equals": 0.0,
        "train/test/identical_sentences_ratio_95": 0.0,
        "train/test/unrelated_sentences_ratio_equals": 98.0,
        "train/test/unrelated_sentences_ratio_5": 98.0,
    }
    run.update(overrides)
    return run


def _dead_run(name: str) -> dict:
    """The collapsed run of the diagnostic: constant output, NaN Pearson, RMSE 73.30."""
    return _run(
        name,
        fold=1,
        summary={"test/st_dev_score": 0.0, "test/pearson_corr": float("nan"), "test/diverged": 1.0},
        **{"test/rmse": 73.30, "test/mean_score": 0.0, "test/st_dev_score": 0.0, "test/pearson_corr": None},
    )


class TestC1Exclusion:
    def test_the_dead_run_never_reaches_the_aggregate(self):
        usable, excluded = filter_usable_runs([_run("alive"), _dead_run("dead")])

        assert [run["name"] for run in usable] == ["alive"]
        assert [run["name"] for run in excluded] == ["dead"]

    def test_the_dead_run_would_have_moved_the_mean(self):
        # Without the filter, the zero-predictor drags the reported RMSE up by 11 points
        # and the predicted mean down by 10, which is how the sweep table was built.
        with_dead = compute_summary_table(group_runs([_run("a"), _dead_run("dead")]))[0]
        usable, _ = filter_usable_runs([_run("a"), _dead_run("dead")])
        without_dead = compute_summary_table(group_runs(usable))[0]

        assert with_dead["RMSE_mean"] > without_dead["RMSE_mean"] + 10
        assert with_dead["Pred mean_mean"] < without_dead["Pred mean_mean"] - 10
        assert without_dead["N_folds"] == 1


class TestC5CompressionColumns:
    def test_predicted_moments_sit_next_to_the_rmse(self):
        row = compute_summary_table(group_runs([_run("a"), _run("b", fold=1)]))[0]

        assert row["Pred mean_mean"] == pytest.approx(21.64)
        assert row["Pred std_mean"] == pytest.approx(9.22)
        assert row["RMSE_mean"] == pytest.approx(51.41)

    def test_compression_factor_is_reported(self):
        row = compute_summary_table(group_runs([_run("a")]))[0]

        assert row["Compression"] == f"{DEFAULT_LABEL_STD / 9.22:.2f}x"
        # The diagnostic reports a compression of 2 to 4 on the four checkpoints.
        assert 2.0 < DEFAULT_LABEL_STD / 9.22 < 5.0

    def test_csv_carries_the_new_columns(self, tmp_path):
        rows = compute_summary_table(group_runs([_run("a"), _run("b", fold=1)]))
        output = tmp_path / "sweep_summary.csv"
        save_csv(rows, str(output))

        with open(output, encoding="utf-8") as handle:
            written = list(csv.DictReader(handle))

        assert "Pred mean" in written[0]
        assert "Pred std" in written[0]
        assert "Compression" in written[0]
        assert written[0]["Pred mean"].startswith("21.64")

    def test_latex_table_carries_the_columns_and_the_label_reference(self, tmp_path):
        rows = compute_summary_table(group_runs([_run("a"), _run("b", fold=1)]))
        output = tmp_path / "sweep_summary.tex"
        generate_latex_table(rows, str(output))
        content = output.read_text(encoding="utf-8")

        assert "Pred mean" in content
        assert "Pred std" in content
        assert f"{DEFAULT_LABEL_MEAN:.2f}" in content
        assert f"{DEFAULT_LABEL_STD:.2f}" in content

    def test_predicted_moments_are_never_bolded_as_a_best_value(self, tmp_path):
        # There is no "best" predicted mean: it is read against the label mean.
        rows = compute_summary_table(
            group_runs(
                [
                    _run("a"),
                    _run("b", checkpoint="microsoft/deberta-v3-large", **{"test/mean_score": 35.68}),
                ]
            )
        )
        output = tmp_path / "sweep_summary.tex"
        generate_latex_table(rows, str(output))
        content = output.read_text(encoding="utf-8")

        assert r"\textbf{35.68" not in content
        assert r"\textbf{21.64" not in content
