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


def _run(variant="d_full", pearson=0.85, rmse=20.0, arch="deberta-v3-base", **overrides):
    condition, _, mode = variant.partition("_")
    fields = {
        "arch": arch,
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


def test_runs_are_found_one_directory_per_architecture(tmp_path):
    """The grid writes results/grid/<arch>-<head>/<variant>.json."""
    for arch in ("bert-clamped", "deberta-clamped"):
        sub = os.path.join(str(tmp_path), arch)
        os.makedirs(sub)
        _write(sub, "d_none", {"test_pearson_corr": 0.8, "test_rmse": 20.0})
    assert len(load_runs(str(tmp_path))) == 2


def test_the_report_decomposes_the_two_diagnostics():
    runs = [
        _run("a_none", pearson=0.80),
        _run("b_none", pearson=0.62),
        _run("c_none", pearson=0.79),
    ]
    report = render(runs)
    assert "fuite par phrase source (H5)" in report
    assert "-0.180" in report  # a -> b, the leak disappearing
    assert "+0.170" in report  # b -> c, the H6 correction


def test_the_report_compares_the_two_corpora_at_equal_augmentation():
    runs = [_run("c_none", pearson=0.79), _run("d_none", pearson=0.84)]
    report = render(runs)
    assert "v1 corrige (c) contre v2 (d)" in report
    assert "+0.050" in report


def test_the_report_compares_augmentation_at_equal_corpus():
    runs = [_run("d_none", pearson=0.80, rmse=22.0), _run("d_full", pearson=0.83, rmse=19.0)]
    report = render(runs)
    assert "Augmentation" in report
    assert "+0.030" in report


def test_the_report_flags_a_diverged_run():
    assert "DIVERGED" in render([_run("d_full", diverged=True)])


def test_the_report_states_the_produit_target():
    report = render([_run("d_full", pearson=0.85, rmse=20.0)])
    assert "0,914" in report


def test_the_objective_multiplies_its_three_terms():
    """A product, so one weak sanity check drags the whole score down."""
    run = _run(pearson=0.90, identical_ratio_95=50.0, unrelated_ratio_5=100.0)
    assert run.objective == pytest.approx(0.45)


def test_the_objective_punishes_a_failing_sanity_check_harder_than_an_average_would():
    strong = _run(pearson=0.90, identical_ratio_95=95.0, unrelated_ratio_5=95.0)
    lopsided = _run(pearson=0.95, identical_ratio_95=20.0, unrelated_ratio_5=100.0)
    assert strong.objective > lopsided.objective


def test_the_objective_is_undefined_when_a_term_is_missing():
    assert math.isnan(_run(pearson=float("nan")).objective)


def test_the_report_names_the_best_configuration_by_objective():
    runs = [
        _run("c_none", pearson=0.95, identical_ratio_95=10.0, unrelated_ratio_5=99.0),
        _run("d_full", pearson=0.85, identical_ratio_95=99.0, unrelated_ratio_5=99.0),
    ]
    report = render(runs)
    assert "Meilleure configuration" in report
    assert report.index("d_full") < report.index("c_none")


def test_the_report_survives_a_partial_experiment():
    """Half the runs finished; the table must still render."""
    assert "a_none" in render([_run("a_none")])


# --- the analyser must survive what actually lives in results/ ------------------------


def test_a_diagnostic_json_is_not_mistaken_for_a_run(tmp_path):
    """results/ also holds host probes and audits. A run carries test metrics; those do not."""
    path = os.path.join(str(tmp_path), "host-renard.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"probe": {"host": "renard"}, "verdict": {"bf16": False}}, handle)
    assert load_run(path) is None


def test_an_audit_json_is_not_mistaken_for_a_run(tmp_path):
    path = os.path.join(str(tmp_path), "calibration_audit.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"label_mean": 62.66, "n_runs": 80, "runs": []}, handle)
    assert load_run(path) is None


def test_a_json_list_at_the_top_level_does_not_crash_the_reader(tmp_path):
    path = os.path.join(str(tmp_path), "whatever.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([1, 2, 3], handle)
    assert load_run(path) is None


def test_a_run_missing_its_row_count_is_read_rather_than_crashing(tmp_path):
    """int(nan) raises, and a crash in the reporting layer would hide every result that
    did survive."""
    path = os.path.join(str(tmp_path), "d_none.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"test": {"test_pearson_corr": 0.8, "test_rmse": 20.0}}, handle)
    run = load_run(path)
    assert run is not None
    assert run.train_rows == 0


def test_diagnostics_and_runs_can_share_a_directory(tmp_path):
    _write(str(tmp_path), "d_none", {"test_pearson_corr": 0.8, "test_rmse": 20.0})
    with open(os.path.join(str(tmp_path), "host-renard.json"), "w", encoding="utf-8") as handle:
        json.dump({"probe": {}}, handle)
    assert [r.variant for r in load_runs(str(tmp_path))] == ["d_none"]
