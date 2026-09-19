"""Tests for the 0-100 harmonisation of heterogeneous corpora."""

import math

import numpy as np
import pytest

from data.harmonize import (
    MIN_ANCHORS_AFFINE,
    MIN_ANCHORS_ISOTONIC,
    AffineAnchorMap,
    Anchors,
    BoundsMap,
    QuantileMap,
    SaturatingCountMap,
    choose_map,
    deduplicate,
    find_anchors,
    fit_affine,
    fit_isotonic,
    fit_quantile,
    harmonize,
    normalise_text,
    pair_key,
)
from data.schema import SCALES, build

REF_LICENSE = "MIT"


def _corpus(name, rows, scale="da100", domain="wiki"):
    """Build a contract dataset from (original, simplification, label_raw) triples."""
    return build(
        [
            {
                "item_id": str(index),
                "original": original,
                "simplification": simplification,
                "label_raw": float(value),
                "scale": scale,
                "domain": domain,
                "license": REF_LICENSE,
            }
            for index, (original, simplification, value) in enumerate(rows)
        ],
        name,
    )


# --- text normalisation --------------------------------------------------------------


def test_normalise_collapses_runs_of_whitespace():
    assert normalise_text("a   b\n\tc ") == "a b c"


def test_normalise_keeps_case_because_case_can_change_meaning():
    assert normalise_text("Apple sells fruit") != normalise_text("apple sells fruit")


def test_pair_key_separates_the_two_sentences_unambiguously():
    """Concatenation without a separator would make ('ab','c') collide with ('a','bc')."""
    assert pair_key("ab", "c") != pair_key("a", "bc")


# --- BoundsMap -----------------------------------------------------------------------


def test_bounds_map_stretches_a_likert5_onto_0_100():
    mapping = BoundsMap(SCALES["likert5"])
    assert mapping([1, 3, 5]) == pytest.approx([0.0, 50.0, 100.0])


def test_bounds_map_inverts_a_scale_where_higher_is_worse():
    """severity3 runs 1=minor to 3=major, so 3 must land on 0, not 100."""
    mapping = BoundsMap(SCALES["severity3"])
    assert mapping([1, 2, 3]) == pytest.approx([100.0, 50.0, 0.0])


def test_bounds_map_is_identity_on_a_scale_already_at_0_100():
    mapping = BoundsMap(SCALES["da100"])
    assert mapping([0, 37.5, 100]) == pytest.approx([0.0, 37.5, 100.0])


def test_bounds_map_refuses_an_unbounded_scale():
    with pytest.raises(ValueError, match="bounded scale"):
        BoundsMap(SCALES["error_count"])


def test_bounds_map_clips_values_that_exceed_the_declared_range():
    mapping = BoundsMap(SCALES["likert5"])
    assert mapping([0.0, 9.0]) == pytest.approx([0.0, 100.0])


# --- SaturatingCountMap --------------------------------------------------------------


def test_zero_errors_means_fully_preserved():
    assert SaturatingCountMap(cap=3)([0]) == pytest.approx([100.0])


def test_the_cap_and_beyond_means_nothing_preserved():
    mapping = SaturatingCountMap(cap=3)
    assert mapping([3, 4, 100]) == pytest.approx([0.0, 0.0, 0.0])


def test_the_count_map_is_linear_between_zero_and_the_cap():
    assert SaturatingCountMap(cap=4)([1, 2, 3]) == pytest.approx([75.0, 50.0, 25.0])


def test_a_non_positive_cap_is_rejected():
    with pytest.raises(ValueError, match="cap must be positive"):
        SaturatingCountMap(cap=0)


# --- anchors -------------------------------------------------------------------------


def test_find_anchors_matches_only_pairs_present_in_both_corpora():
    reference = _corpus("csmd", [("a", "A", 80.0), ("b", "B", 20.0)])
    source = _corpus("other", [("a", "A", 4.0), ("zzz", "ZZZ", 1.0)], scale="likert5")
    anchors = find_anchors(source, reference)
    assert len(anchors) == 1
    assert anchors.raw == pytest.approx([4.0])
    assert anchors.reference == pytest.approx([80.0])


def test_find_anchors_ignores_whitespace_differences_between_corpora():
    reference = _corpus("csmd", [("a  b", "c", 80.0)])
    source = _corpus("other", [("a b", "c", 4.0)], scale="likert5")
    assert len(find_anchors(source, reference)) == 1


def test_find_anchors_averages_repeated_scores_on_each_side():
    reference = _corpus("csmd", [("a", "A", 80.0), ("a", "A", 60.0)])
    source = _corpus("other", [("a", "A", 4.0), ("a", "A", 2.0)], scale="likert5")
    anchors = find_anchors(source, reference)
    assert len(anchors) == 1
    assert anchors.reference == pytest.approx([70.0])
    assert anchors.raw == pytest.approx([3.0])


def test_no_shared_pair_yields_no_anchor():
    reference = _corpus("csmd", [("a", "A", 80.0)])
    source = _corpus("other", [("q", "Q", 4.0)], scale="likert5")
    assert len(find_anchors(source, reference)) == 0


def test_anchor_pearson_is_undefined_when_one_side_is_constant():
    anchors = Anchors(raw=np.ones(10), reference=np.arange(10, dtype=float))
    assert math.isnan(anchors.pearson)


# --- fitting -------------------------------------------------------------------------


def _linear_anchors(n, slope=25.0, intercept=-25.0):
    raw = np.linspace(1, 5, n)
    return Anchors(raw=raw, reference=slope * raw + intercept)


def test_affine_fit_recovers_a_known_linear_relation():
    mapping = fit_affine(_linear_anchors(40))
    assert mapping.slope == pytest.approx(25.0, abs=1e-6)
    assert mapping.intercept == pytest.approx(-25.0, abs=1e-6)
    assert mapping([1, 3, 5]) == pytest.approx([0.0, 50.0, 100.0])


def test_affine_fit_refuses_too_few_anchors():
    with pytest.raises(ValueError, match="at least"):
        fit_affine(_linear_anchors(MIN_ANCHORS_AFFINE - 1))


def test_affine_fit_refuses_constant_raw_scores():
    anchors = Anchors(raw=np.full(40, 3.0), reference=np.linspace(0, 100, 40))
    with pytest.raises(ValueError, match="unidentifiable"):
        fit_affine(anchors)


def test_isotonic_fit_refuses_too_few_anchors():
    with pytest.raises(ValueError, match="at least"):
        fit_isotonic(_linear_anchors(MIN_ANCHORS_ISOTONIC - 1))


def test_isotonic_fit_is_monotone_on_a_non_linear_relation():
    raw = np.linspace(1, 5, 200)
    anchors = Anchors(raw=raw, reference=100 * ((raw - 1) / 4) ** 2)
    mapping = fit_isotonic(anchors)
    out = mapping(np.linspace(1, 5, 50))
    assert np.all(np.diff(out) >= -1e-9)


def test_quantile_map_lands_inside_the_reference_range():
    mapping = fit_quantile(np.linspace(0, 10, 200), np.linspace(30, 90, 200))
    out = mapping([0, 5, 10])
    assert out.min() >= 30 - 1e-6
    assert out.max() <= 90 + 1e-6


def test_every_map_clips_into_0_100():
    mapping = AffineAnchorMap(slope=1000.0, intercept=-5000.0, n_anchors=50, anchor_pearson=0.9)
    out = mapping([0.0, 100.0])
    assert out.min() >= 0.0
    assert out.max() <= 100.0


# --- choose_map ----------------------------------------------------------------------


def _shared(n, scale="likert5"):
    """A reference and a source sharing *n* pairs with a known linear relation."""
    rows_ref = [(f"o{i}", f"s{i}", 25.0 * (1 + 4 * i / max(1, n - 1)) - 25.0) for i in range(n)]
    rows_src = [(f"o{i}", f"s{i}", 1 + 4 * i / max(1, n - 1)) for i in range(n)]
    return _corpus("csmd", rows_ref), _corpus("other", rows_src, scale=scale)


def test_choose_map_prefers_isotonic_when_anchors_are_plentiful():
    reference, source = _shared(MIN_ANCHORS_ISOTONIC + 10)
    mapping, anchors, _ = choose_map(source, reference, count_cap=3.0)
    assert mapping.name == "isotonic_anchor"
    assert len(anchors) >= MIN_ANCHORS_ISOTONIC


def test_choose_map_falls_back_to_affine_and_says_so():
    reference, source = _shared(MIN_ANCHORS_AFFINE + 2)
    mapping, _, messages = choose_map(source, reference, count_cap=3.0)
    assert mapping.name == "affine_anchor"
    assert any("non-linearity" in m for m in messages)


def test_choose_map_falls_back_to_bounds_with_a_loud_warning_when_no_anchor():
    reference = _corpus("csmd", [("a", "A", 50.0)])
    source = _corpus("other", [("q", "Q", 3.0)], scale="likert5")
    mapping, anchors, messages = choose_map(source, reference, count_cap=3.0)
    assert mapping.name == "bounds"
    assert len(anchors) == 0
    assert any("weakest link" in m for m in messages)


def test_choose_map_reports_anchors_it_had_to_discard_as_too_few():
    reference, source = _shared(3)
    _, anchors, messages = choose_map(source, reference, count_cap=3.0)
    assert len(anchors) == 3
    assert any("below the" in m for m in messages)


def test_choose_map_uses_the_saturating_map_for_unbounded_counts():
    reference = _corpus("csmd", [("a", "A", 50.0)])
    source = _corpus("other", [("q", "Q", 2.0)], scale="error_count")
    mapping, _, messages = choose_map(source, reference, count_cap=5.0)
    assert mapping.name == "saturating_count"
    assert any("cap of 5.0" in m for m in messages)


def test_choose_map_never_picks_quantile_on_its_own():
    """Distribution matching assumes comparable system quality; it must be opt-in."""
    reference = _corpus("csmd", [("a", "A", 50.0)])
    source = _corpus("other", [("q", "Q", 3.0)], scale="likert5")
    mapping, _, _ = choose_map(source, reference, count_cap=3.0)
    assert not isinstance(mapping, QuantileMap)


# --- harmonize -----------------------------------------------------------------------


def test_harmonize_copies_the_reference_labels_unchanged():
    reference = _corpus("csmd", [("a", "A", 80.0), ("b", "B", 20.0)])
    merged, report = harmonize({"csmd": reference}, reference="csmd")
    assert merged["label"] == pytest.approx([80.0, 20.0])
    assert report.mappings[0].mapping == "identity"


def test_harmonize_fills_every_label_and_leaves_none_nan():
    reference, source = _shared(MIN_ANCHORS_AFFINE + 5)
    with pytest.warns(UserWarning):
        merged, _ = harmonize({"csmd": reference, "other": source})
    assert not any(math.isnan(value) for value in merged["label"])


def test_harmonize_keeps_every_row_of_every_corpus():
    reference, source = _shared(MIN_ANCHORS_AFFINE + 5)
    with pytest.warns(UserWarning):
        merged, _ = harmonize({"csmd": reference, "other": source})
    assert len(merged) == len(reference) + len(source)


def test_harmonize_rejects_a_missing_reference():
    with pytest.raises(KeyError, match="missing"):
        harmonize({"other": _corpus("other", [("a", "A", 3.0)], scale="likert5")}, reference="csmd")


def test_harmonize_rejects_a_reference_that_is_not_on_da100():
    bad = _corpus("csmd", [("a", "A", 3.0)], scale="likert5")
    with pytest.raises(ValueError, match="da100"):
        harmonize({"csmd": bad}, reference="csmd")


def test_harmonize_honours_an_explicit_override():
    reference, source = _shared(MIN_ANCHORS_ISOTONIC + 10)
    override = BoundsMap(SCALES["likert5"])
    merged, report = harmonize({"csmd": reference, "other": source}, overrides={"other": override})
    assert [m.mapping for m in report.mappings if m.corpus == "other"] == ["bounds"]
    assert len(merged) == len(reference) + len(source)


def test_the_report_names_the_corpora_placed_without_any_anchor():
    reference = _corpus("csmd", [("a", "A", 50.0)])
    source = _corpus("other", [("q", "Q", 3.0)], scale="likert5")
    with pytest.warns(UserWarning):
        _, report = harmonize({"csmd": reference, "other": source})
    assert [m.corpus for m in report.weakest] == ["other"]


def test_the_report_summary_mentions_every_corpus():
    reference, source = _shared(MIN_ANCHORS_AFFINE + 5)
    with pytest.warns(UserWarning):
        _, report = harmonize({"csmd": reference, "other": source})
    summary = report.summary()
    assert "csmd" in summary and "other" in summary


# --- deduplication -------------------------------------------------------------------


def _merged_for_dedup():
    a = _corpus("csmd", [("shared", "S", 80.0), ("only_a", "A", 10.0)])
    b = _corpus("other", [("shared", "S", 40.0), ("only_b", "B", 30.0)])
    from datasets import concatenate_datasets

    a = a.remove_columns(["label"]).add_column("label", [80.0, 10.0])
    b = b.remove_columns(["label"]).add_column("label", [40.0, 30.0])
    return concatenate_datasets([a, b.select_columns(a.column_names)])


def test_dedup_keeps_the_higher_priority_copy_of_a_shared_pair():
    kept, dropped = deduplicate(_merged_for_dedup(), priority=["csmd", "other"])
    shared = [row for row in kept if row["original"] == "shared"]
    assert len(shared) == 1
    assert shared[0]["corpus"] == "csmd"
    assert dropped == {"other": 1}


def test_dedup_priority_order_actually_decides_the_winner():
    kept, _ = deduplicate(_merged_for_dedup(), priority=["other", "csmd"])
    shared = [row for row in kept if row["original"] == "shared"]
    assert shared[0]["corpus"] == "other"


def test_dedup_keeps_pairs_unique_to_one_corpus():
    kept, _ = deduplicate(_merged_for_dedup(), priority=["csmd", "other"])
    assert {row["original"] for row in kept} == {"shared", "only_a", "only_b"}


def test_a_corpus_absent_from_the_priority_list_ranks_last():
    kept, _ = deduplicate(_merged_for_dedup(), priority=["other"])
    shared = [row for row in kept if row["original"] == "shared"]
    assert shared[0]["corpus"] == "other"


def test_dedup_is_idempotent():
    once, _ = deduplicate(_merged_for_dedup(), priority=["csmd", "other"])
    twice, dropped = deduplicate(once, priority=["csmd", "other"])
    assert len(once) == len(twice)
    assert dropped == {}


# --- likert3_signed ------------------------------------------------------------------


def test_the_signed_likert_maps_minus_one_to_zero_and_one_to_one_hundred():
    mapping = BoundsMap(SCALES["likert3_signed"])
    assert mapping([-1.0, 0.0, 1.0]) == pytest.approx([0.0, 50.0, 100.0])


def test_the_signed_likert_is_not_confused_with_severity_three():
    """severity3 shares the cardinality and inverts; conflating them would flip the sign."""
    signed = BoundsMap(SCALES["likert3_signed"])([1.0])
    severity = BoundsMap(SCALES["severity3"])([3.0])
    assert signed == pytest.approx([100.0])
    assert severity == pytest.approx([0.0])
