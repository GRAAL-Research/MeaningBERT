"""Tests for the v2 experiment: corpus builder and result analysis."""

import json
import math
import os

import pytest

from data.build_corpus import CONDITIONS, _v1_row_split, build_condition
from data.schema import build
from figures_generator.analyze_v2_experiment import LABEL_STD, RunResult, load_run, load_runs, render


def _dataset(rows):
    built = build(
        [
            {
                "item_id": str(index),
                "original": original,
                "simplification": simplification,
                "label_raw": 70.0,
                "scale": "da100",
                "domain": "wiki",
                "license": "MIT",
                "source": source,
            }
            for index, (original, simplification, source) in enumerate(rows)
        ],
        "demo",
    )
    return built.remove_columns(["label"]).add_column("label", [70.0] * len(built))


# --- the v1 row split, kept on purpose --------------------------------------------


def _v1_input(n=60):
    rows = []
    for i in range(n):
        rows.append((f"sentence {i}", f"simplification {i}", "original"))
        rows.append((f"sentence {i}", f"other simplification {i}", "original"))
        rows.append((f"sentence {i}", f"sentence {i}", "identical"))
        rows.append((f"sentence {i}", f"unrelated {i}", "unrelated"))
    return _dataset(rows)


def test_the_v1_split_produces_the_four_expected_splits():
    splits = _v1_row_split(_v1_input(), seed=42)
    assert set(splits) == {"train", "dev", "test", "sanity"}


def test_the_v1_split_leaks_source_sentences_which_is_the_point():
    """Condition 'a' must reproduce the defect, not a cleaned-up version of it."""
    splits = _v1_row_split(_v1_input(), seed=42)
    train = {row["original"] for row in splits["train"]}
    test = {row["original"] for row in splits["test"]}
    assert len(train & test) > 0


def test_the_v1_sanity_split_is_drawn_from_the_test_pool():
    splits = _v1_row_split(_v1_input(), seed=42)
    assert set(splits["sanity"]["source"]) <= {"identical", "unrelated"}
    test_ids = set(splits["test"]["item_id"])
    assert set(splits["sanity"]["item_id"]) <= test_ids


def test_the_v1_split_keeps_every_row():
    dataset = _v1_input(20)
    splits = _v1_row_split(dataset, seed=1)
    assert len(splits["train"]) + len(splits["dev"]) + len(splits["test"]) == len(dataset)


def test_the_v1_split_is_deterministic():
    first = _v1_row_split(_v1_input(20), seed=3)["train"]["item_id"]
    second = _v1_row_split(_v1_input(20), seed=3)["train"]["item_id"]
    assert first == second


def test_an_unknown_condition_is_refused():
    with pytest.raises(ValueError, match="unknown condition"):
        build_condition("z", seed=42)


def test_the_ladder_has_exactly_four_rungs():
    assert CONDITIONS == ("a", "b", "c", "d")


# --- result analysis ---------------------------------------------------------------


def _run(variant="d_full", pearson=0.85, rmse=20.0, **overrides):
    condition, _, mode = variant.partition("_")
    fields = {
        "variant": variant,
        "condition": condition,
        "mode": mode,
        "pearson": pearson,
        "rmse": rmse,
        "r2": 0.7,
        "pred_mean": 62.0,
        "pred_std": 30.0,
        "identical_ratio_95": 92.0,
        "unrelated_ratio_5": 97.0,
        "epochs": 30.0,
        "train_rows": 3000,
        "diverged": False,
    }
    fields.update(overrides)
    return RunResult(**fields)


def test_the_rmse_floor_follows_the_closed_form():
    run = _run(pearson=0.8)
    assert run.rmse_floor == pytest.approx(LABEL_STD * math.sqrt(1 - 0.64), abs=1e-6)


def test_a_perfect_correlation_leaves_no_floor():
    assert _run(pearson=1.0).rmse_floor == pytest.approx(0.0)


def test_a_missing_correlation_gives_no_floor_rather_than_crashing():
    assert math.isnan(_run(pearson=float("nan")).rmse_floor)


def test_the_floor_is_symmetric_in_the_sign_of_the_correlation():
    assert _run(pearson=-0.8).rmse_floor == pytest.approx(_run(pearson=0.8).rmse_floor)


def _write(tmp_path, variant, test_metrics, identical=None, unrelated=None):
    payload = {
        "run_name": variant,
        "epochs_trained": 12,
        "rows": {"train": 1000},
        "test": test_metrics,
        "identical": identical or {},
        "unrelated": unrelated or {},
    }
    path = os.path.join(tmp_path, f"{variant}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def test_a_run_json_is_read_into_a_result(tmp_path):
    path = _write(str(tmp_path), "c_none", {"test_pearson_corr": 0.81, "test_rmse": 19.0})
    run = load_run(path)
    assert run is not None
    assert run.condition == "c"
    assert run.mode == "none"
    assert run.pearson == pytest.approx(0.81)


def test_an_unreadable_file_is_skipped_rather_than_fatal(tmp_path):
    path = os.path.join(str(tmp_path), "broken.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert load_run(path) is None


def test_a_missing_metric_becomes_nan_not_zero(tmp_path):
    """Zero would silently look like a real, very bad score."""
    path = _write(str(tmp_path), "a_none", {"test_rmse": 40.0})
    run = load_run(path)
    assert math.isnan(run.pearson)


def test_a_string_valued_metric_does_not_crash_the_reader(tmp_path):
    path = _write(str(tmp_path), "a_none", {"test_pearson_corr": "NaN", "test_rmse": 40.0})
    assert load_run(path) is not None


def test_runs_are_ordered_by_rung_then_by_augmentation(tmp_path):
    for variant in ("d_full", "a_none", "c_full", "a_full"):
        _write(str(tmp_path), variant, {"test_pearson_corr": 0.8, "test_rmse": 20.0})
    assert [r.variant for r in load_runs(str(tmp_path))] == ["a_none", "a_full", "c_full", "d_full"]


def test_the_report_decomposes_each_rung_of_the_ladder():
    runs = [
        _run("a_none", pearson=0.80),
        _run("b_none", pearson=0.62),
        _run("c_none", pearson=0.79),
        _run("d_none", pearson=0.84),
    ]
    report = render(runs)
    assert "source-sentence leak (H5)" in report
    assert "-0.180" in report  # a -> b, the leak disappearing
    assert "+0.170" in report  # b -> c, the H6 correction
    assert "+0.050" in report  # c -> d, the added corpora


def test_the_report_compares_augmentation_at_each_rung():
    runs = [_run("d_none", pearson=0.80, rmse=22.0), _run("d_full", pearson=0.83, rmse=19.0)]
    report = render(runs)
    assert "+0.030" in report
    assert "-3.00" in report


def test_the_report_flags_a_diverged_run():
    assert "DIVERGED" in render([_run("d_full", diverged=True)])


def test_the_report_states_the_produit_target():
    report = render([_run("d_full", pearson=0.85, rmse=20.0)])
    assert "0.914" in report
    assert "RMSE < 15" in report


def test_the_report_survives_a_partial_experiment():
    """Half the runs finished; the table must still render."""
    assert "a_none" in render([_run("a_none")])
