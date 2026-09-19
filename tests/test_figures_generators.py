"""Tests for the C5 columns of the article table.

``python2latex`` is not importable in every environment (0.4.1 ships a broken
``from version import __version__``), and building a real LaTeX document needs a LaTeX
installation. The table object is therefore stubbed: what is under test is which wandb
summary keys feed which cell, not how python2latex renders them.
"""

import sys
import types

import pytest

from diagnostics.calibration_audit import DEFAULT_LABEL_MEAN, DEFAULT_LABEL_STD


class _TableStub:
    """Records the cells that get written, so a test can read them back."""

    def __init__(self, n_rows: int) -> None:
        self.cells: dict[tuple[int, int], object] = {}
        self.n_rows = n_rows

    def __setitem__(self, key, value) -> None:
        row, col = key
        self.cells[(row % self.n_rows, col)] = value


@pytest.fixture(name="figures_generators", scope="module")
def figures_generators_fixture():
    """Import ``figures_generators`` against a stubbed ``python2latex``."""
    saved_p2l = sys.modules.get("python2latex")
    saved_module = sys.modules.pop("figures_generator.figures_generators", None)

    stub = types.ModuleType("python2latex")
    stub.Document = object
    stub.Table = object
    stub.italic = lambda value: value
    sys.modules["python2latex"] = stub

    import figures_generator.figures_generators as module  # pylint: disable=import-outside-toplevel

    yield module

    sys.modules.pop("figures_generator.figures_generators", None)
    if saved_module is not None:
        sys.modules["figures_generator.figures_generators"] = saved_module
    if saved_p2l is not None:
        sys.modules["python2latex"] = saved_p2l
    else:
        sys.modules.pop("python2latex", None)


def _benchmark_summaries() -> list[dict]:
    """Two runs carrying the per-metric moments logged by ``evaluate_metrics``."""
    return [
        {"test/BLEU_mean": 30.0, "test/BLEU_st_dev": 20.0, "test/SARI_mean": 40.0, "test/SARI_st_dev": 12.0},
        {"test/BLEU_mean": 34.0, "test/BLEU_st_dev": 22.0, "test/SARI_mean": 44.0, "test/SARI_st_dev": 14.0},
    ]


def _meaningbert_summaries(mean_score: float, st_dev_score: float) -> list[dict]:
    """Two MeaningBERT runs with the compressed moments the sweep reported."""
    return [
        {"test/mean_score": mean_score, "test/st_dev_score": st_dev_score},
        {"test/mean_score": mean_score + 2.0, "test/st_dev_score": st_dev_score + 1.0},
    ]


def _few_shot_data() -> list:
    """The five groups ``get_table_1115`` expects."""
    return [
        _meaningbert_summaries(21.64, 9.22),  # without data augmentation
        _meaningbert_summaries(35.68, 16.88),  # with data augmentation
        [],
        _benchmark_summaries(),
        [],
    ]


class TestMomentColumn:
    def test_benchmark_rows_are_filled_from_the_metric_keys(self, figures_generators):
        table = _TableStub(24)

        figures_generators._fill_moment_column(  # pylint: disable=protected-access
            table,
            col_idx=4,
            few_shot_data=_few_shot_data(),
            metric_key=lambda metric: f"test/{metric}_mean",
            meaning_bert_key="test/mean_score",
            metrics=["BLEU", "SARI"],
        )

        assert table.cells[(1, 4)] == "32.00" + r"$\pm$" + "2.83"
        assert table.cells[(2, 4)] == "42.00" + r"$\pm$" + "2.83"

    def test_meaningbert_rows_carry_the_compressed_moments(self, figures_generators):
        table = _TableStub(24)

        figures_generators._fill_moment_column(  # pylint: disable=protected-access
            table,
            col_idx=5,
            few_shot_data=_few_shot_data(),
            metric_key=lambda metric: f"test/{metric}_st_dev",
            meaning_bert_key="test/st_dev_score",
            metrics=["BLEU", "SARI"],
        )

        # Without DA: spreads 9.22 and 10.22, against a label spread of 37.01.
        assert table.cells[(22, 5)] == "9.72" + r"$\pm$" + "0.71"
        assert table.cells[(23, 5)] == "17.38" + r"$\pm$" + "0.71"

    def test_a_missing_key_does_not_crash_the_table(self, figures_generators):
        table = _TableStub(24)

        figures_generators._fill_moment_column(  # pylint: disable=protected-access
            table,
            col_idx=4,
            few_shot_data=[[{}], [{}], [], [{}], []],
            metric_key=lambda metric: f"test/{metric}_mean",
            meaning_bert_key="test/mean_score",
            metrics=["BLEU"],
        )

        assert table.cells[(1, 4)] == "n/a"
        assert table.cells[(22, 4)] == "n/a"

    def test_the_caption_states_the_label_distribution(self, figures_generators):
        caption = figures_generators.label_reference_caption()

        assert f"{DEFAULT_LABEL_MEAN:.2f}" in caption
        assert f"{DEFAULT_LABEL_STD:.2f}" in caption
