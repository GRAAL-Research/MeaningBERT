"""Tests for the source-sentence-grouped splitter.

The guarantee under test is the one CSMD v1 lacks: no source sentence on two sides of the
wall. See ``docs/H5-fuite-par-phrase-source.md``.
"""

import pytest
from datasets import Dataset, DatasetDict

from data.schema import build
from data.splits import (
    DEFAULT_ALLOWED_GROUP_OVERLAPS,
    LeakageError,
    assert_no_leakage,
    group_key,
    split_by_source_sentence,
)


def _dataset(rows):
    """Build a harmonised dataset from (original, simplification, source) triples."""
    built = build(
        [
            {
                "item_id": str(index),
                "original": original,
                "simplification": simplification,
                "label_raw": 50.0,
                "scale": "da100",
                "domain": "wiki",
                "license": "MIT",
                "source": source,
            }
            for index, (original, simplification, source) in enumerate(rows)
        ],
        "demo",
    )
    return built.remove_columns(["label"]).add_column("label", [50.0] * len(built))


def _many_groups(n_groups=40, per_group=4, with_sanity=True):
    """A corpus with *n_groups* source sentences, each carrying *per_group* rows."""
    rows = []
    for g in range(n_groups):
        original = f"source sentence number {g}"
        for k in range(per_group):
            rows.append((original, f"simplification {g}.{k}", "original"))
        if with_sanity:
            rows.append((original, original, "identical"))
            rows.append((original, f"totally unrelated word soup {g}", "unrelated"))
    return _dataset(rows)


# --- the core guarantee --------------------------------------------------------------


def test_no_source_sentence_reaches_both_train_and_test():
    splits, _ = split_by_source_sentence(_many_groups(), seed=1)
    train = {group_key(o) for o in splits["train"]["original"]}
    test = {group_key(o) for o in splits["test"]["original"]}
    assert train & test == set()


def test_no_source_sentence_reaches_both_train_and_dev():
    splits, _ = split_by_source_sentence(_many_groups(), seed=1)
    train = {group_key(o) for o in splits["train"]["original"]}
    dev = {group_key(o) for o in splits["dev"]["original"]}
    assert train & dev == set()


def test_the_sanity_holdout_never_touches_train():
    splits, _ = split_by_source_sentence(_many_groups(), seed=1)
    train = {group_key(o) for o in splits["train"]["original"]}
    sanity = {group_key(o) for o in splits["sanity"]["original"]}
    assert train & sanity == set()


def test_the_produced_split_passes_its_own_leakage_check():
    splits, _ = split_by_source_sentence(_many_groups(), seed=3)
    assert_no_leakage(splits)


def test_the_guarantee_holds_across_many_seeds():
    for seed in range(12):
        splits, _ = split_by_source_sentence(_many_groups(n_groups=25), seed=seed)
        assert_no_leakage(splits)


def test_every_row_lands_in_exactly_one_split():
    dataset = _many_groups(n_groups=20)
    splits, report = split_by_source_sentence(dataset, seed=5)
    assert sum(report.rows.values()) + report.dropped_duplicate_pairs == len(dataset)
    ids = [i for name in splits for i in splits[name]["item_id"]]
    assert len(ids) == len(set(ids))


# --- deduplication -------------------------------------------------------------------


def test_exact_duplicate_pairs_are_dropped_before_splitting():
    dataset = _dataset([("a", "A", "original"), ("a", "A", "original"), ("b", "B", "original")])
    _, report = split_by_source_sentence(dataset, seed=0)
    assert report.dropped_duplicate_pairs == 1


def test_duplicates_differing_only_by_whitespace_are_still_duplicates():
    dataset = _dataset([("a  b", "A", "original"), ("a b", "A", "original")])
    _, report = split_by_source_sentence(dataset, seed=0)
    assert report.dropped_duplicate_pairs == 1


def test_the_same_source_with_different_simplifications_is_not_a_duplicate():
    dataset = _dataset([("a", "A1", "original"), ("a", "A2", "original")])
    _, report = split_by_source_sentence(dataset, seed=0)
    assert report.dropped_duplicate_pairs == 0


# --- sanity holdout ------------------------------------------------------------------


def test_the_sanity_split_holds_only_identical_and_unrelated_rows():
    splits, _ = split_by_source_sentence(_many_groups(), seed=2)
    assert set(splits["sanity"]["source"]) <= {"identical", "unrelated"}


def test_a_zero_sanity_fraction_leaves_the_sanity_split_empty():
    splits, report = split_by_source_sentence(_many_groups(), seed=2, sanity_frac=0.0)
    assert report.rows["sanity"] == 0
    assert len(splits["sanity"]) == 0


def test_reserving_everything_keeps_all_sanity_rows_out_of_train():
    splits, _ = split_by_source_sentence(_many_groups(), seed=2, sanity_frac=1.0)
    assert "identical" not in set(splits["train"]["source"])
    assert "unrelated" not in set(splits["train"]["source"])


def test_a_corpus_without_sanity_rows_still_splits():
    splits, report = split_by_source_sentence(_many_groups(with_sanity=False), seed=2)
    assert report.rows["sanity"] == 0
    assert report.rows["train"] > 0
    assert_no_leakage(splits)


# --- proportions ---------------------------------------------------------------------


def test_row_proportions_land_near_their_targets():
    dataset = _many_groups(n_groups=200, per_group=3, with_sanity=False)
    _, report = split_by_source_sentence(dataset, seed=7, dev_frac=0.1, test_frac=0.2, sanity_frac=0.0)
    total = sum(report.rows.values())
    assert report.rows["test"] / total == pytest.approx(0.20, abs=0.03)
    assert report.rows["dev"] / total == pytest.approx(0.10, abs=0.03)


def test_wildly_uneven_group_sizes_still_hit_the_targets():
    """SimpEval carries ~40 simplifications per source; proportional sampling would miss."""
    rows = []
    for g in range(60):
        size = 40 if g < 5 else 2
        for k in range(size):
            rows.append((f"src {g}", f"simp {g}.{k}", "original"))
    _, report = split_by_source_sentence(_dataset(rows), seed=11, sanity_frac=0.0)
    total = sum(report.rows.values())
    assert report.rows["test"] / total == pytest.approx(0.20, abs=0.10)
    assert report.rows["train"] > report.rows["test"]


def test_splitting_is_deterministic_for_a_given_seed():
    dataset = _many_groups(n_groups=30)
    first, _ = split_by_source_sentence(dataset, seed=9)
    second, _ = split_by_source_sentence(dataset, seed=9)
    assert first["train"]["item_id"] == second["train"]["item_id"]


def test_different_seeds_produce_different_splits():
    dataset = _many_groups(n_groups=30)
    first, _ = split_by_source_sentence(dataset, seed=9)
    second, _ = split_by_source_sentence(dataset, seed=10)
    assert first["train"]["item_id"] != second["train"]["item_id"]


# --- argument validation -------------------------------------------------------------


def test_fractions_that_leave_no_room_for_train_are_rejected():
    with pytest.raises(ValueError, match="leave room for train"):
        split_by_source_sentence(_many_groups(), dev_frac=0.5, test_frac=0.5)


def test_an_out_of_range_sanity_fraction_is_rejected():
    with pytest.raises(ValueError, match=r"sanity_frac must be in \[0, 1\]"):
        split_by_source_sentence(_many_groups(), sanity_frac=1.5)


# --- assert_no_leakage as a detector -------------------------------------------------


def _leaky_splits():
    shared = _dataset([("a", "A1", "original")])
    other = _dataset([("a", "A2", "original")])
    return DatasetDict({"train": shared, "test": other})


def test_the_checker_catches_a_shared_source_sentence():
    with pytest.raises(LeakageError, match="share 1 source sentence"):
        assert_no_leakage(_leaky_splits())


def test_the_checker_catches_an_exact_duplicate_pair():
    same = _dataset([("a", "A", "original")])
    with pytest.raises(LeakageError, match="exact pair"):
        assert_no_leakage(DatasetDict({"train": same, "test": same}))


def test_the_checker_reports_the_offending_sentence():
    with pytest.raises(LeakageError, match="'a'"):
        assert_no_leakage(_leaky_splits())


def test_test_and_sanity_may_share_a_source_sentence_by_design():
    """A reserved group feeds sanity and test on purpose; neither is a training set."""
    pairs = DatasetDict({"test": _dataset([("a", "A1", "original")]), "sanity": _dataset([("a", "a", "identical")])})
    assert_no_leakage(pairs)


def test_that_allowance_does_not_extend_to_train():
    pairs = DatasetDict({"train": _dataset([("a", "A1", "original")]), "sanity": _dataset([("a", "a", "identical")])})
    with pytest.raises(LeakageError):
        assert_no_leakage(pairs)


def test_the_allowance_can_be_revoked():
    pairs = DatasetDict({"test": _dataset([("a", "A1", "original")]), "sanity": _dataset([("a", "a", "identical")])})
    with pytest.raises(LeakageError):
        assert_no_leakage(pairs, allowed_group_overlaps=frozenset())


def test_the_default_allowance_is_exactly_test_and_sanity():
    assert DEFAULT_ALLOWED_GROUP_OVERLAPS == frozenset({frozenset({"test", "sanity"})})


def test_a_clean_split_raises_nothing():
    clean = DatasetDict({"train": _dataset([("a", "A", "original")]), "test": _dataset([("b", "B", "original")])})
    assert_no_leakage(clean)
