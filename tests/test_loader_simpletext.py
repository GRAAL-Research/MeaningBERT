"""Tests for the SimpleText loader. Runs entirely on local fixtures, no network access."""

# pylint: disable=redefined-outer-name
# Pytest fixtures are conventionally re-injected as same-named test parameters.

from __future__ import annotations

import math
import statistics
from pathlib import Path

import pytest
from datasets import Dataset

from data.loaders.simpletext import load
from data.schema import validate

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "simpletext"


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load(raw_dir=FIXTURES_DIR)


def _row_by_snt_id(dataset: Dataset, snt_id_suffix: str) -> dict:
    matches = [row for row in dataset if row["item_id"].endswith(snt_id_suffix)]
    assert len(matches) == 1, f"expected exactly one row ending in {snt_id_suffix!r}, found {len(matches)}"
    return matches[0]


def test_validate_passes_on_loader_output(dataset: Dataset) -> None:
    validate(dataset)  # raises ContractError on any violation


def test_malformed_row_with_missing_simplification_is_dropped(dataset: Dataset) -> None:
    # G01.1_1019677957_2 has an empty ("simplified sentence") field in the source CSV.
    item_ids = [row["item_id"] for row in dataset]
    assert not any(item_id.endswith("1019677957_2") for item_id in item_ids)


def test_fluency_only_error_does_not_count_as_meaning_loss(dataset: Dataset) -> None:
    # G01.1_1019677957_3 is flagged "No error" = False in the source, but only for a
    # fluency/alignment issue; no C or D taxonomy flag is set, so meaning is preserved.
    row = _row_by_snt_id(dataset, "1019677957_3")
    assert row["label_raw"] == 0.0
    assert row["n_annotators"] == 1
    assert math.isnan(row["label_std"])


def test_two_agreeing_annotators_yield_zero_std(dataset: Dataset) -> None:
    # G01.1_147704292_9: both annotators E and B recorded zero meaning errors.
    row = _row_by_snt_id(dataset, "147704292_9")
    assert row["n_annotators"] == 2
    assert row["label_raw"] == 0.0
    assert row["label_std"] == 0.0


def test_two_disagreeing_annotators_are_averaged(dataset: Dataset) -> None:
    # G01.1_147704292_2: annotator E flags 2 meaning errors (overgeneralization + loss of
    # informative content), annotator B flags 1 (loss of informative content only).
    row = _row_by_snt_id(dataset, "147704292_2")
    assert row["n_annotators"] == 2
    assert row["label_raw"] == pytest.approx(statistics.mean([2.0, 1.0]))
    assert row["label_std"] == pytest.approx(statistics.stdev([2.0, 1.0]))


def test_self_consistency_duplicate_collapses_to_one_annotator(dataset: Dataset) -> None:
    # T15.1_1576337284_8: annotator B appears once in test_data.csv (0 meaning errors) and
    # once more in inter_annotator_agreement.csv (2 meaning errors) for the SAME pair - a
    # self-consistency rerun in the source study, not a second annotator. The loader must
    # average B's two judgments into a single value and count 6 annotators, not 7.
    row = _row_by_snt_id(dataset, "T15.1_1576337284_8")
    per_annotator_values = {
        "A": 1.0,  # D1.1 Overgeneralization
        "D": 0.0,
        "C": 2.0,  # C2 Faithfulness hallucination + D1.2 Overspecification
        "F": 1.0,  # D1.2 Overspecification
        "B": statistics.mean([0.0, 2.0]),  # collapsed self-consistency rerun
        "E": 0.0,
    }
    assert row["n_annotators"] == 6
    assert row["label_raw"] == pytest.approx(statistics.mean(per_annotator_values.values()))
    assert row["label_std"] == pytest.approx(statistics.stdev(per_annotator_values.values()))


def test_item_id_is_stable_across_runs() -> None:
    first = {row["item_id"] for row in load(raw_dir=FIXTURES_DIR)}
    second = {row["item_id"] for row in load(raw_dir=FIXTURES_DIR)}
    assert first == second
    assert len(first) > 0


def test_item_id_is_prefixed_and_unique(dataset: Dataset) -> None:
    item_ids = [row["item_id"] for row in dataset]
    assert len(item_ids) == len(set(item_ids))
    assert all(item_id.startswith("simpletext:") for item_id in item_ids)


def test_scale_and_domain_are_fixed(dataset: Dataset) -> None:
    assert set(dataset["scale"]) == {"error_count"}
    assert set(dataset["domain"]) == {"scientific"}
    assert set(dataset["corpus"]) == {"simpletext"}
    assert all(value >= 0.0 for value in dataset["label_raw"])


def test_system_carries_the_run_id(dataset: Dataset) -> None:
    row = _row_by_snt_id(dataset, "147704292_9")
    assert row["system"] == "FRANE_AND_ANDREA_Task3.1_t5"


def test_missing_raw_files_raise_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing raw SimpleText file"):
        load(raw_dir=tmp_path)
