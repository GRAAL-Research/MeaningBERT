"""Tests for the module that produces every number the v3 paper reports.

``src/figures_generator/analyse_v3.py`` is the only place the grid is turned into prose,
so a silent error here does not crash anything: it prints a wrong number into a table and
the paper carries it. The tests therefore target the three places a wrong number can come
from without raising -- reading a cell from the wrong directory, reading a class from the
wrong row of a confusion matrix, and formatting a mean away from its standard deviation --
rather than the shape of the LaTeX.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from figures_generator.analyse_v3 import (
    as_percent,
    cohen_d,
    error,
    figure,
    load,
    main_table,
    paired,
    stats_table,
    suite_recalls,
    suites_table,
    welch,
)

CLASSES = ["entailment", "neutral", "contradiction"]


def write_cell(root: str, arch: str, condition: str, seed: int, **scores) -> None:
    """Write one ``metrics.json`` the way a training run does."""
    path = os.path.join(root, "polarity", f"{arch}-{condition}", f"seed{seed}")
    os.makedirs(path)
    payload = {
        "test": {"accuracy": scores.get("accuracy", 0.5), "macro_f1": scores["macro_f1"]},
        "probes": {
            "sanity": {
                "accuracy": scores["sanity"],
                "class_names": CLASSES,
                "confusion": scores.get("confusion", [[10, 0, 0], [0, 10, 0], [0, 0, 10]]),
            },
            "nan_nli": {"accuracy": scores.get("nan_nli", 0.5)},
            "monli": {"accuracy": scores.get("monli", 0.5)},
        },
    }
    with open(os.path.join(path, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class TestLoad:
    def test_architecture_name_is_split_from_the_right(self, tmp_path):
        """``nli-deberta-v3-base-none`` is one architecture and one condition, not five.

        Splitting from the left reads the architecture as ``nli``, which would merge two
        different encoders into one row of the table without failing.
        """
        root = str(tmp_path)
        write_cell(root, "nli-deberta-v3-base", "none", 42, macro_f1=0.9, sanity=0.8)
        write_cell(root, "deberta-v3-base", "none", 42, macro_f1=0.1, sanity=0.2)

        cells = load(root)

        assert set(cells) == {("nli-deberta-v3-base", "none"), ("deberta-v3-base", "none")}
        assert cells[("nli-deberta-v3-base", "none")]["macro_f1"] == [0.9]

    def test_unknown_architecture_is_ignored(self, tmp_path):
        """A directory that is not in the paper's registry must not reach a table."""
        root = str(tmp_path)
        write_cell(root, "some-other-encoder", "none", 42, macro_f1=0.9, sanity=0.8)

        assert load(root) == {}

    def test_every_seed_of_a_cell_is_collected(self, tmp_path):
        root = str(tmp_path)
        for seed, value in ((42, 0.90), (43, 0.92)):
            write_cell(root, "bert", "full", seed, macro_f1=value, sanity=0.8)

        assert sorted(load(root)[("bert", "full")]["macro_f1"]) == [0.90, 0.92]


class TestSuiteRecalls:
    def test_recall_is_read_by_class_name_not_by_position(self):
        """The confusion matrix is indexed through ``class_names``.

        A checkpoint whose head is in a different order would otherwise report the
        identical suite's accuracy against the mirrored suite's row.
        """
        sanity = {
            "class_names": ["contradiction", "neutral", "entailment"],
            "confusion": [[6, 2, 2], [0, 8, 2], [0, 0, 10]],
        }

        got = suite_recalls(sanity)

        assert got["identical"] == pytest.approx(1.0)
        assert got["unrelated"] == pytest.approx(0.8)
        assert got["mirrored"] == pytest.approx(0.6)

    def test_an_empty_row_gives_nan_rather_than_dividing_by_zero(self):
        sanity = {"class_names": CLASSES, "confusion": [[0, 0, 0], [0, 5, 0], [0, 0, 5]]}

        assert math.isnan(suite_recalls(sanity)["identical"])


class TestStatistics:
    def test_cohen_d_is_the_difference_over_the_pooled_spread(self):
        left, right = [2.0, 4.0, 6.0], [1.0, 3.0, 5.0]

        assert cohen_d(left, right) == pytest.approx(1.0 / 2.0)

    def test_cohen_d_refuses_a_single_observation(self):
        assert math.isnan(cohen_d([1.0], [2.0, 3.0]))

    def test_welch_reports_the_difference_in_the_order_given(self):
        got = welch([0.9, 0.91, 0.92], [0.80, 0.81, 0.82])

        assert got["diff"] == pytest.approx(0.1, abs=1e-9)
        assert got["p"] < 0.05

    def test_welch_sees_no_effect_in_identical_samples(self):
        got = welch([0.9, 0.91, 0.92], [0.9, 0.91, 0.92])

        assert got["diff"] == pytest.approx(0.0)
        assert got["p"] == pytest.approx(1.0)

    def test_paired_refuses_unequal_lengths(self):
        """Pairing by seed is only meaningful when both sides ran the same seeds."""
        assert math.isnan(paired([1.0, 2.0, 3.0], [1.0, 2.0])["diff"])

    def test_paired_removes_the_variance_the_two_sides_share(self):
        """A constant offset over noisy seeds is invisible to Welch and obvious to pairing."""
        left = [0.10, 0.50, 0.90, 0.30]
        right = [value - 0.02 for value in left]

        assert paired(left, right)["diff"] == pytest.approx(0.02)
        assert paired(left, right)["p"] < 0.001
        assert welch(left, right)["p"] > 0.5

    def test_error_is_the_standard_error_of_the_difference(self):
        left, right = [1.0, 2.0, 3.0], [4.0, 6.0, 8.0]
        expected = math.sqrt(1.0 / 3 + 4.0 / 3)

        assert error(left, right) == pytest.approx(expected)


class TestFormatting:
    def test_a_mean_is_reported_as_a_percentage_beside_its_spread(self):
        assert as_percent([0.90, 0.92]) == r"91.00\,$\pm$\,1.41"

    def test_a_missing_cell_prints_a_dash_rather_than_a_zero(self):
        """An absent run must not be read as a model that scored nothing."""
        assert as_percent([]) == "--"

    def test_a_single_seed_reports_a_zero_spread(self):
        assert as_percent([0.5]) == r"50.00\,$\pm$\,0.00"


@pytest.fixture(name="grid")
def grid_fixture(tmp_path):
    """Two encoders, two conditions, two seeds, with one deliberate best cell."""
    root = str(tmp_path)
    table = {
        ("bert", "none"): (0.70, 0.60),
        ("bert", "full"): (0.72, 0.98),
        ("deberta-v3-large", "none"): (0.90, 0.61),
        ("deberta-v3-large", "full"): (0.88, 0.99),
    }
    for (arch, condition), (task, sanity) in table.items():
        for seed, bump in ((42, 0.0), (43, 0.01)):
            write_cell(root, arch, condition, seed, macro_f1=task + bump, sanity=sanity)
    return load(root)


class TestTables:
    def test_the_best_value_of_a_column_is_the_one_in_bold(self, grid):
        """The caption promises bold marks the best value, so it must sit on that row."""
        rows = [line for line in main_table(grid).splitlines() if r"\textsc{raw}" in line or r"\textsc{aug}" in line]
        bolded = [line for line in rows if r"\textbf{90" in line or r"\textbf{91" in line]

        assert len(bolded) == 1
        assert "DeBERTa-v3-large" in bolded[0] and r"\textsc{raw}" in bolded[0]

    def test_only_the_encoders_that_have_runs_get_a_row(self, grid):
        """Seven encoders are declared and two are present; the five absent ones

        must not appear as a row of dashes, which would read as five models that
        scored nothing rather than five models that were not run.
        """
        rows = [line for line in main_table(grid).splitlines() if line.endswith(r"\\")]
        data = [line for line in rows if r"\textsc{raw}" in line or r"\textsc{aug}" in line]

        assert len(data) == 4
        assert not any("--" in line for line in data)

    def test_the_dagger_marks_the_differences_that_are_not_significant(self, grid):
        """Sanity moves by tens of points here and the task barely moves at all."""
        lines = stats_table(grid).splitlines()
        bert = next(line for line in lines if line.startswith("BERT-base"))

        assert r"$^{\dagger}$" in bert
        assert "+38.00" in bert

    def test_the_suite_table_reports_the_three_families_as_percentages(self, grid):
        body = suites_table(grid)

        assert "Identical" in body and "Unrelated" in body and "Mirrored" in body
        assert r"100.00\,$\pm$\,0.00" in body


class TestFigure:
    def test_the_figure_carries_no_legend_and_no_grid(self, grid, tmp_path):
        """The conditions are named in the caption, and the panels keep their ticks only."""
        path = str(tmp_path / "fig.tex")

        figure(grid, path)
        with open(path, encoding="utf-8") as handle:
            body = handle.read()

        assert r"\legend" not in body
        assert "xmajorgrids=false" in body and "ymajorgrids=false" in body

    def test_the_points_are_plotted_in_percent(self, grid, tmp_path):
        """The axes say percent, so the coordinates must not still be in the unit interval."""
        path = str(tmp_path / "fig.tex")

        figure(grid, path)
        with open(path, encoding="utf-8") as handle:
            body = handle.read()

        assert "(70.50," in body
        assert "(0.70," not in body


class TestAxisRange:
    def test_the_range_covers_every_point_and_its_error_bar(self, grid):
        """A hard-coded range drops an outlying point without any compiler saying so."""
        from figures_generator.analyse_v3 import axis_range

        low, high = axis_range(grid, ["bert", "deberta-v3-large"], "macro_f1")

        assert low <= 70.0
        assert high >= 90.5

    def test_the_range_stops_just_past_a_full_hundred_percent(self, grid):
        from figures_generator.analyse_v3 import axis_range

        assert axis_range(grid, ["bert", "deberta-v3-large"], "sanity")[1] <= 101.0

    def test_an_empty_panel_falls_back_to_the_full_scale(self):
        from figures_generator.analyse_v3 import axis_range

        assert axis_range({}, ["bert"], "macro_f1") == (0, 100)


class TestDegenerateCells:
    def test_a_single_seed_puts_a_nan_in_the_table_rather_than_stopping_the_build(self):
        """One run of a cell has no variance; the other six encoders must still print."""
        assert math.isnan(error([0.9], [0.8, 0.81]))


class TestUnlabelledCells:
    """A trained cell nobody declared would vanish from every table without a word."""

    def test_a_tag_absent_from_the_registry_is_reported(self, tmp_path):
        from figures_generator.analyse_v3 import unlabelled_cells

        root = str(tmp_path)
        write_cell(root, "bert", "none", 42, macro_f1=0.9, sanity=0.8)
        write_cell(root, "some-new-encoder", "none", 42, macro_f1=0.9, sanity=0.8)

        assert unlabelled_cells(root) == {"some-new-encoder"}

    def test_a_fully_declared_grid_reports_nothing(self, tmp_path):
        from figures_generator.analyse_v3 import unlabelled_cells

        root = str(tmp_path)
        write_cell(root, "bert", "full", 42, macro_f1=0.9, sanity=0.8)

        assert unlabelled_cells(root) == set()

    def test_a_directory_that_is_not_a_cell_is_not_reported(self, tmp_path):
        """The weight-keeping runs use their own condition suffix and are not grid cells."""
        from figures_generator.analyse_v3 import unlabelled_cells

        root = str(tmp_path)
        write_cell(root, "nli-deberta-v3-large", "none-poids", 42, macro_f1=0.9, sanity=0.8)

        assert unlabelled_cells(root) == set()
