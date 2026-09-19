"""Tests for the PLABA loader.

The loader is intentionally blocked (see ``src/data/loaders/plaba.py`` and ``RAPPORT.md``):
the native annotation scale of the TREC PLABA manual accuracy judgments does not appear in
CONTRACT.md's SCALES table, so ``load()`` raises instead of returning a dataset. These tests
therefore exercise the real parsing and aggregation logic (which is fully implemented) and
verify that the block itself is deliberate and clearly reported, not a silent failure.

Fixtures under ``tests/fixtures/plaba/`` are small, real excerpts (a handful of rows each)
copied verbatim from https://github.com/ondovb/plaba-ft (retrieved 2026-09-19), no network
access at test time.
"""

from pathlib import Path

import pytest

from data.loaders.plaba import UnsupportedScaleError, extract_rows, load

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "plaba" / "eval" / "manual"


def test_extract_rows_reads_every_fixture_file():
    result = extract_rows(FIXTURES_DIR)
    # 6 + 6 + 6 + 2 rows across the four fixture files.
    assert result.n_rows_read == 20


def test_extract_rows_keeps_faithfulness_on_native_scale():
    result = extract_rows(FIXTURES_DIR)
    labels = [row["label_raw"] for row in result.rows]
    assert all(label in (-1.0, 0.0, 1.0) for label in labels)


def test_extract_rows_drops_rows_with_empty_output_and_counts_them():
    # tests/fixtures/plaba/eval/manual/Manual-3.acc.csv holds two real rows where the
    # annotators evaluated an omitted sentence: Output is empty, score is -1 for both axes.
    result = extract_rows(FIXTURES_DIR)
    assert result.n_rows_dropped_empty_text == 2
    kept_from_manual_3 = [row for row in result.rows if row["item_id"].startswith("manual-3:")]
    assert kept_from_manual_3 == []


def test_extract_rows_item_id_is_stable_across_two_runs():
    first = extract_rows(FIXTURES_DIR)
    second = extract_rows(FIXTURES_DIR)
    assert [row["item_id"] for row in first.rows] == [row["item_id"] for row in second.rows]


def test_extract_rows_item_ids_are_unique():
    result = extract_rows(FIXTURES_DIR)
    item_ids = [row["item_id"] for row in result.rows]
    assert len(item_ids) == len(set(item_ids))


def test_extract_rows_assigns_human_system_to_manual_files():
    result = extract_rows(FIXTURES_DIR)
    manual_rows = [row for row in result.rows if row["item_id"].startswith("manual-1:")]
    assert manual_rows and all(row["system"] == "human" for row in manual_rows)


def test_extract_rows_assigns_named_system_to_model_files():
    result = extract_rows(FIXTURES_DIR)
    gpt_rows = [row for row in result.rows if row["item_id"].startswith("gpt-3.5-zero_shot:")]
    assert gpt_rows and all(row["system"] == "gpt-3.5-zero-shot" for row in gpt_rows)


def test_extract_rows_deduplicates_identical_pairs_within_a_system():
    result = extract_rows(FIXTURES_DIR)
    keys = [(row["original"], row["simplification"], row["system"]) for row in result.rows]
    assert len(keys) == len(set(keys))


def test_extract_rows_raises_on_missing_raw_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_rows(tmp_path / "does-not-exist")


def test_extract_rows_raises_on_unrecognised_system_file(tmp_path):
    unknown = tmp_path / "SomeNewTeam.acc.csv"
    unknown.write_text("Abst,Sent,Source,Output,Acc. comp.,Acc. faith.\nQ1_A1,1,src,out,1,1\n", encoding="utf-8")
    with pytest.raises(UnsupportedScaleError):
        extract_rows(tmp_path)


def test_extract_rows_raises_on_malformed_header(tmp_path):
    malformed = tmp_path / "GPT-3.5-zero_shot.acc.csv"
    malformed.write_text("Abst,Sent,Source,Output\nQ1_A1,1,src,out\n", encoding="utf-8")
    with pytest.raises(UnsupportedScaleError):
        extract_rows(tmp_path)


def test_load_raises_unsupported_scale_error_naming_the_native_bounds(monkeypatch):
    import data.loaders.plaba as plaba_module

    monkeypatch.setattr(plaba_module, "RAW_DIR", FIXTURES_DIR)
    with pytest.raises(UnsupportedScaleError, match=r"\[-1\.0, 1\.0\]"):
        load()
