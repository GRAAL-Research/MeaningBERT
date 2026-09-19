"""Tests for the SimpEval loader (``src/data/loaders/simpeval.py``).

Runs entirely on local fixtures under ``tests/fixtures/simpeval/``: no network. The
fixtures are small extracts of the real ``simpeval_past.csv`` and ``simpDA_2022.csv``,
deliberately chosen to cover a malformed row, an identical-text row/group and a normal
multi-annotator row/group.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from datasets import Dataset

from data.loaders.simpeval import _dedupe_within_corpus, _load_2022_rows, _load_past_rows
from data.schema import build, validate

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "simpeval"
PAST_FIXTURE = FIXTURES_DIR / "simpeval_past.csv"
DA2022_FIXTURE = FIXTURES_DIR / "simpDA_2022.csv"


# --- simpeval_past.csv ----------------------------------------------------------------


def test_load_past_rows_drops_the_row_with_an_empty_generation():
    rows, counters = _load_past_rows(PAST_FIXTURE)
    assert all(row["item_id"] != "past-1682" for row in rows)
    assert counters["empty_simplification"] == 1


def test_load_past_rows_marks_a_literally_identical_pair_as_identical():
    rows, _ = _load_past_rows(PAST_FIXTURE)
    copy_row = next(row for row in rows if row["item_id"] == "past-14")
    assert copy_row["system"] == "copy"
    assert copy_row["source"] == "identical"


def test_load_past_rows_marks_a_genuine_simplification_as_original():
    rows, _ = _load_past_rows(PAST_FIXTURE)
    lstm_row = next(row for row in rows if row["item_id"] == "past-0")
    assert lstm_row["source"] == "original"


def test_load_past_rows_averages_the_five_annotator_ratings():
    rows, _ = _load_past_rows(PAST_FIXTURE)
    row = next(row for row in rows if row["item_id"] == "past-0")
    # ratings 73, 88, 95, 85, 87
    assert row["label_raw"] == pytest.approx(85.6)
    assert row["label_std"] == pytest.approx(7.9875, abs=1e-3)
    assert row["n_annotators"] == 5
    assert row["scale"] == "da100"
    assert row["domain"] == "wiki"


def test_load_past_rows_uses_processed_generation_not_raw_generation():
    rows, _ = _load_past_rows(PAST_FIXTURE)
    row = next(row for row in rows if row["item_id"] == "past-0")
    # the raw "generation" field is lowercased/detokenized; processed_generation is not.
    assert (
        row["simplification"]
        == "In late 2004, Suleman made headlines by cutting Howard Stern's radio show from four Citadel stations."
    )


def test_load_past_rows_drops_a_row_with_a_non_numeric_rating(tmp_path):
    bad_csv = tmp_path / "bad_past.csv"
    header = (
        "id,original,generation,processed_generation,original_id,system,rating_1,rating_2,rating_3,rating_4,rating_5\n"
    )
    bad_row = '0,"orig","gen","gen",0,sys,not_a_number,88,95,85,87\n'
    bad_csv.write_text(header + bad_row, encoding="utf-8")

    rows, counters = _load_past_rows(bad_csv)
    assert rows == []
    assert counters["malformed_ratings"] == 1


def test_load_past_rows_item_id_is_stable_across_two_runs():
    first, _ = _load_past_rows(PAST_FIXTURE)
    second, _ = _load_past_rows(PAST_FIXTURE)
    assert [row["item_id"] for row in first] == [row["item_id"] for row in second]


# --- simpDA_2022.csv --------------------------------------------------------------------


def test_load_2022_rows_aggregates_adequacy_across_the_three_workers():
    rows, _ = _load_2022_rows(DA2022_FIXTURE)
    row = next(row for row in rows if row["item_id"] == "2022-40-GPT-3-few-shot")
    # Answer.adequacy: 100, 91, 91
    assert row["label_raw"] == pytest.approx(94.0)
    assert row["label_std"] == pytest.approx(5.19615, abs=1e-4)
    assert row["n_annotators"] == 3
    assert row["scale"] == "da100"


def test_load_2022_rows_marks_an_identical_group_from_all_three_workers():
    rows, _ = _load_2022_rows(DA2022_FIXTURE)
    row = next(row for row in rows if row["item_id"] == "2022-39-T5-3B")
    assert row["source"] == "identical"
    assert row["label_raw"] == pytest.approx(100.0)
    assert row["label_std"] == pytest.approx(0.0)


def test_load_2022_rows_preserves_a_multi_word_system_name_verbatim():
    rows, _ = _load_2022_rows(DA2022_FIXTURE)
    row = next(row for row in rows if row["item_id"] == "2022-11-Human_1_Writing")
    assert row["system"] == "Human 1 Writing"
    assert row["label_raw"] == pytest.approx(71.66667, abs=1e-4)


def test_load_2022_rows_drops_a_group_with_empty_source_text(tmp_path):
    bad_csv = tmp_path / "bad_da2022.csv"
    header = "WorkerId,Input.id,Input.original,Input.simplified,Input.system,Answer.adequacy\n"
    rows_text = '0,1,,"some text",sysA,80\n1,1,,"some text",sysA,85\n2,1,,"some text",sysA,90\n'
    bad_csv.write_text(header + rows_text, encoding="utf-8")

    rows, counters = _load_2022_rows(bad_csv)
    assert rows == []
    assert counters["empty_text"] == 3


def test_load_2022_rows_std_is_nan_for_a_lone_annotator(tmp_path):
    single_csv = tmp_path / "single_da2022.csv"
    header = "WorkerId,Input.id,Input.original,Input.simplified,Input.system,Answer.adequacy\n"
    row = "0,1,orig,simp,sysA,80\n"
    single_csv.write_text(header + row, encoding="utf-8")

    rows, _ = _load_2022_rows(single_csv)
    assert len(rows) == 1
    assert rows[0]["n_annotators"] == 1
    assert math.isnan(rows[0]["label_std"])


# --- dedup ------------------------------------------------------------------------------


def test_dedupe_within_corpus_drops_a_repeated_triple():
    row = {"original": "a", "simplification": "b", "system": "sysA", "item_id": "x"}
    duplicate = dict(row, item_id="y")
    kept, n_dropped = _dedupe_within_corpus([row, duplicate])
    assert kept == [row]
    assert n_dropped == 1


def test_dedupe_within_corpus_keeps_the_same_pair_from_different_systems():
    row_a = {"original": "a", "simplification": "b", "system": "sysA", "item_id": "x"}
    row_b = {"original": "a", "simplification": "b", "system": "sysB", "item_id": "y"}
    kept, n_dropped = _dedupe_within_corpus([row_a, row_b])
    assert kept == [row_a, row_b]
    assert n_dropped == 0


# --- end-to-end against the contract -----------------------------------------------------


def test_fixture_rows_pass_the_contract_validator():
    past_rows, _ = _load_past_rows(PAST_FIXTURE)
    da2022_rows, _ = _load_2022_rows(DA2022_FIXTURE)
    rows, _ = _dedupe_within_corpus(past_rows + da2022_rows)

    dataset = build(rows, corpus="simpeval")
    validate(dataset)  # must not raise

    assert isinstance(dataset, Dataset)
    assert set(dataset["scale"]) == {"da100"}
    assert set(dataset["corpus"]) == {"simpeval"}
    assert all(item_id.startswith("simpeval:") for item_id in dataset["item_id"])


def test_fixture_item_ids_are_prefixed_per_subset():
    past_rows, _ = _load_past_rows(PAST_FIXTURE)
    da2022_rows, _ = _load_2022_rows(DA2022_FIXTURE)
    dataset = build(past_rows + da2022_rows, corpus="simpeval")

    assert any(item_id.startswith("simpeval:past-") for item_id in dataset["item_id"])
    assert any(item_id.startswith("simpeval:2022-") for item_id in dataset["item_id"])
