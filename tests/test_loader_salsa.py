"""Tests for the SALSA edit-level loader.

SALSA has no network access in CI: every test reads ``tests/fixtures/salsa/salsa_sample.json``,
a small extract (7 sentence pairs) of the real interface-demo data served at
``https://thresh.tools/data/salsa.json``. See ``RAPPORT.md`` for why the loader stops at the
edit level instead of emitting a CONTRACT-compliant dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.loaders.salsa import (
    EditRecord,
    MalformedEdit,
    SalsaAggregationPending,
    _pair_id,
    _parse_deletion,
    _parse_insertion,
    _parse_reorder,
    _parse_split,
    _parse_structure,
    _parse_substitution,
    _unwrap,
    candidate_aggregations,
    classify_family,
    load,
    load_edit_level,
    parse_item_edits,
    skip_reasons,
)

FIXTURE = Path(__file__).parent / "fixtures" / "salsa" / "salsa_sample.json"


# --- load(): the contract boundary must stay closed until arbitrated ------------------


def test_load_refuses_to_guess_an_aggregation():
    with pytest.raises(SalsaAggregationPending):
        load()


# --- _unwrap: the recurring {val: name, name: rest} nesting ---------------------------


def test_unwrap_returns_the_named_branch_and_its_payload():
    assert _unwrap({"val": "good", "good": "a lot"}) == ("good", "a lot")


def test_unwrap_returns_none_payload_for_a_bare_leaf():
    assert _unwrap({"val": "trivial"}) == ("trivial", None)


def test_unwrap_rejects_a_node_without_a_val_key():
    with pytest.raises(MalformedEdit):
        _unwrap({"good": "a lot"})


# --- Per-category parsing: real shapes taken from interface/salsa.yml -----------------


def test_parse_deletion_good_carries_severity_and_less_information():
    parsed = _parse_deletion(
        {"deletion_type": {"val": "good_deletion", "good_deletion": "a lot"}, "grammar_error": "no"}
    )
    assert parsed["quality"] == "good"
    assert parsed["severity"] == 3
    assert parsed["information_impact"] == "less"


def test_parse_deletion_trivial_has_no_severity():
    parsed = _parse_deletion({"deletion_type": {"val": "trivial_deletion"}, "grammar_error": "no"})
    assert parsed["quality"] == "trivial"
    assert parsed["severity"] is None


def test_parse_deletion_flags_coreference_error():
    parsed = _parse_deletion(
        {
            "deletion_type": {"val": "bad_deletion", "bad_deletion": "somewhat"},
            "coreference": "yes",
            "grammar_error": "no",
        }
    )
    assert parsed["coreference_error"] is True


def test_parse_deletion_rejects_an_unknown_subtype():
    with pytest.raises(MalformedEdit):
        _parse_deletion({"deletion_type": {"val": "mystery_deletion"}, "grammar_error": "no"})


def test_parse_insertion_elaboration_is_good_and_adds_information():
    parsed = _parse_insertion({"insertion_type": {"val": "elaboration", "elaboration": "minor"}, "grammar_error": "no"})
    assert parsed["quality"] == "good"
    assert parsed["severity"] == 1
    assert parsed["information_impact"] == "more"


def test_parse_insertion_trivial_insertion_no_is_trivial():
    parsed = _parse_insertion(
        {"insertion_type": {"val": "trivial_insertion", "trivial_insertion": "no"}, "grammar_error": "no"}
    )
    assert parsed["quality"] == "trivial"
    assert parsed["severity"] is None


def test_parse_insertion_trivial_insertion_yes_is_good_with_a_rating():
    # Not observed in the 50-item demo (every real trivial_insertion answers "no"), but
    # interface/salsa.yml documents a "yes" branch with its own likert-3 follow-up
    # question, nested the same way every other yes/no-then-rate branch is. Constructed
    # here to exercise that documented, schema-driven path, not to stand in for corpus data.
    parsed = _parse_insertion(
        {
            "insertion_type": {"val": "trivial_insertion", "trivial_insertion": {"val": "yes", "yes": "somewhat"}},
            "grammar_error": "no",
        }
    )
    assert parsed["quality"] == "good"
    assert parsed["severity"] == 2


def test_parse_insertion_error_subtype_is_bad():
    parsed = _parse_insertion(
        {"insertion_type": {"val": "contradiction", "contradiction": "a lot"}, "grammar_error": "no"}
    )
    assert parsed["quality"] == "bad"
    assert parsed["severity"] == 3


def test_parse_substitution_same_trivial_is_a_no_op_paraphrase():
    parsed = _parse_substitution(
        {"substitution_info_change": {"val": "same", "same": {"val": "trivial"}}, "grammar_error": "no"}
    )
    assert parsed["quality"] == "trivial"
    assert parsed["information_impact"] == "same"


def test_parse_substitution_less_bad_deletion_is_a_content_error():
    parsed = _parse_substitution(
        {
            "substitution_info_change": {"val": "less", "less": {"val": "bad_deletion", "bad_deletion": "somewhat"}},
            "grammar_error": "yes",
        }
    )
    assert parsed["quality"] == "bad"
    assert parsed["severity"] == 2
    assert parsed["information_impact"] == "less"


def test_parse_substitution_different_is_always_an_error_regardless_of_rating():
    # Mirrors process_diff_info in the upstream dataloader: a meaning-changing
    # substitution is always Quality.ERROR, the rating only sets its severity.
    parsed = _parse_substitution(
        {"substitution_info_change": {"val": "different", "different": "minor"}, "grammar_error": "no"}
    )
    assert parsed["quality"] == "bad"
    assert parsed["severity"] == 1
    assert parsed["information_impact"] == "different"


def test_parse_reorder_component_level_good():
    parsed = _parse_reorder(
        {
            "reorder_level": {"val": "component_level", "component_level": {"val": "good", "good": "somewhat"}},
            "grammar_error": "no",
        }
    )
    assert parsed["quality"] == "good"
    assert parsed["reorder_level"] == "component_level"
    assert parsed["information_impact"] == "same"


def test_parse_reorder_rejects_an_unknown_level():
    with pytest.raises(MalformedEdit):
        _parse_reorder({"reorder_level": {"val": "sentence_level"}, "grammar_error": "no"})


def test_parse_structure_carries_its_structure_type():
    parsed = _parse_structure(
        {"structure_type": {"val": "voice"}, "impact": {"val": "good", "good": "a lot"}, "grammar_error": "no"}
    )
    assert parsed["structure_type"] == "voice"
    assert parsed["quality"] == "good"
    assert parsed["severity"] == 3


def test_parse_split_has_no_structure_type():
    # Not present in the 50-item demo (no split edits were annotated there), but split and
    # structure share the exact same {impact: {...}} shape per interface/salsa.yml.
    parsed = _parse_split({"impact": {"val": "bad", "bad": "minor"}, "grammar_error": "no"})
    assert parsed["structure_type"] is None
    assert parsed["quality"] == "bad"
    assert parsed["severity"] == 1


# --- Family classification and the meaning-relevance filter ---------------------------


def test_classify_family_content_when_information_changes_and_edit_is_not_trivial():
    assert classify_family("deletion", "bad", "less") == "content"


def test_classify_family_lexical_for_a_substitution_that_keeps_information():
    assert classify_family("substitution", "good", "same") == "lexical"


def test_classify_family_lexical_for_any_trivial_edit():
    assert classify_family("reorder", "trivial", "same") == "lexical"


def test_classify_family_syntax_for_a_non_trivial_reorder():
    assert classify_family("reorder", "bad", "same") == "syntax"


def test_meaning_relevant_content_edit_always_counts():
    records, _ = parse_item_edits(
        {
            "source": "a",
            "target": "b",
            "metadata": {"system": "sys", "annotator": "ann"},
            "edits": [
                {
                    "category": "deletion",
                    "annotation": {
                        "deletion_type": {"val": "bad_deletion", "bad_deletion": "a lot"},
                        "grammar_error": "no",
                    },
                }
            ],
        },
        0,
    )
    assert records[0].family == "content"
    assert records[0].meaning_relevant is True


def test_meaning_irrelevant_syntax_edit_never_counts_even_when_bad():
    records, _ = parse_item_edits(
        {
            "source": "a",
            "target": "b",
            "metadata": {"system": "sys", "annotator": "ann"},
            "edits": [
                {
                    "category": "structure",
                    "annotation": {
                        "structure_type": {"val": "voice"},
                        "impact": {"val": "bad", "bad": "a lot"},
                        "grammar_error": "no",
                    },
                }
            ],
        },
        0,
    )
    assert records[0].family == "syntax"
    assert records[0].meaning_relevant is False


def test_meaning_irrelevant_trivial_paraphrase_never_counts():
    records, _ = parse_item_edits(
        {
            "source": "a",
            "target": "b",
            "metadata": {"system": "sys", "annotator": "ann"},
            "edits": [
                {
                    "category": "substitution",
                    "annotation": {
                        "substitution_info_change": {"val": "same", "same": {"val": "trivial"}},
                        "grammar_error": "no",
                    },
                }
            ],
        },
        0,
    )
    assert records[0].meaning_relevant is False


def test_meaning_relevant_bad_paraphrase_counts_even_without_information_change():
    records, _ = parse_item_edits(
        {
            "source": "a",
            "target": "b",
            "metadata": {"system": "sys", "annotator": "ann"},
            "edits": [
                {
                    "category": "substitution",
                    "annotation": {
                        "substitution_info_change": {"val": "same", "same": {"val": "bad", "bad": "somewhat"}},
                        "grammar_error": "no",
                    },
                }
            ],
        },
        0,
    )
    assert records[0].family == "lexical"
    assert records[0].meaning_relevant is True


# --- Malformed-line handling: skip and record why, never crash the whole item ---------


def test_parse_item_edits_skips_an_unknown_category_and_keeps_the_rest():
    item = {
        "source": "a",
        "target": "b",
        "metadata": {"system": "sys", "annotator": "ann"},
        "edits": [
            {"category": "teleportation", "annotation": {}},
            {
                "category": "deletion",
                "annotation": {"deletion_type": {"val": "trivial_deletion"}, "grammar_error": "no"},
            },
        ],
    }
    records, reasons = parse_item_edits(item, 0)
    assert len(records) == 1
    assert records[0].category == "deletion"
    assert any("unknown category" in reason for reason in reasons)


def test_parse_item_edits_skips_a_malformed_edit_and_keeps_the_rest():
    item = {
        "source": "a",
        "target": "b",
        "metadata": {"system": "sys", "annotator": "ann"},
        "edits": [
            {
                "category": "deletion",
                "annotation": {"deletion_type": {"val": "not_a_real_subtype"}, "grammar_error": "no"},
            },
            {
                "category": "deletion",
                "annotation": {"deletion_type": {"val": "trivial_deletion"}, "grammar_error": "no"},
            },
        ],
    }
    records, reasons = parse_item_edits(item, 3)
    assert len(records) == 1
    assert any("item 3 edit 0" in reason for reason in reasons)


def test_parse_item_edits_skips_a_pair_with_an_empty_sentence():
    item = {"source": "  ", "target": "something", "metadata": {"system": "sys", "annotator": "ann"}, "edits": []}
    records, reasons = parse_item_edits(item, 5)
    assert not records
    assert any("empty" in reason for reason in reasons)


# --- item_id / pair_id stability -------------------------------------------------------


def test_pair_id_is_deterministic_across_calls():
    first = _pair_id("The cat sat.", "The cat sat.", "human")
    second = _pair_id("The cat sat.", "The cat sat.", "human")
    assert first == second


def test_pair_id_differs_when_the_system_differs():
    a = _pair_id("The cat sat.", "The cat is seated.", "human")
    b = _pair_id("The cat sat.", "The cat is seated.", "gpt-3")
    assert a != b


def test_item_id_is_stable_across_two_independent_loads_of_the_same_fixture():
    first = {record.item_id for record in _edit_records_from_fixture()}
    second = {record.item_id for record in _edit_records_from_fixture()}
    assert first == second
    assert len(first) > 0


def _edit_records_from_fixture() -> list[EditRecord]:
    with FIXTURE.open(encoding="utf-8") as handle:
        raw_items = json.load(handle)
    records: list[EditRecord] = []
    for index, item in enumerate(raw_items):
        item_records, _ = parse_item_edits(item, index)
        records.extend(item_records)
    return records


# --- Multiple annotators on the same sentence pair -------------------------------------


def test_fixture_contains_a_pair_annotated_by_two_different_annotators():
    records = _edit_records_from_fixture()
    pairs_to_annotators: dict[str, set[str]] = {}
    for record in records:
        pairs_to_annotators.setdefault(record.pair_id, set()).add(record.annotator)
    multi_annotator_pairs = [pair for pair, annotators in pairs_to_annotators.items() if len(annotators) > 1]
    assert multi_annotator_pairs, "fixture should keep at least one multi-annotator pair to exercise this case"


def test_multiple_annotators_on_the_same_pair_get_distinct_item_ids_sharing_one_pair_id():
    records = _edit_records_from_fixture()
    pairs_to_annotators: dict[str, set[str]] = {}
    for record in records:
        pairs_to_annotators.setdefault(record.pair_id, set()).add(record.item_id)
    multi = [item_ids for item_ids in pairs_to_annotators.values() if len(item_ids) > 1]
    assert multi
    for item_ids in multi:
        assert len(item_ids) == len({item_id.split(":")[-1] for item_id in item_ids})


# --- load_edit_level(): the actual deliverable, against the local fixture -------------


def test_load_edit_level_reads_from_the_local_fixture_without_network():
    dataset = load_edit_level(raw_path=FIXTURE)
    assert len(dataset) > 0
    assert set(dataset.column_names) == set(EditRecord.__dataclass_fields__)


def test_load_edit_level_only_emits_known_categories_and_families():
    dataset = load_edit_level(raw_path=FIXTURE)
    assert set(dataset["category"]) <= {"deletion", "insertion", "substitution", "reorder", "split", "structure"}
    assert set(dataset["family"]) <= {"content", "syntax", "lexical"}


def test_load_edit_level_severity_is_always_within_likert_3_bounds_or_none():
    dataset = load_edit_level(raw_path=FIXTURE)
    for severity in dataset["severity"]:
        assert severity is None or 1 <= severity <= 3


def test_skip_reasons_is_empty_on_the_well_formed_fixture():
    assert not skip_reasons(raw_path=FIXTURE)


# --- candidate_aggregations(): illustrative only, never applied by load() -------------


def test_candidate_aggregations_are_none_or_zero_with_no_content_errors():
    records, _ = parse_item_edits(
        {
            "source": "a",
            "target": "b",
            "metadata": {"system": "sys", "annotator": "ann"},
            "edits": [
                {
                    "category": "substitution",
                    "annotation": {
                        "substitution_info_change": {"val": "same", "same": {"val": "trivial"}},
                        "grammar_error": "no",
                    },
                }
            ],
        },
        0,
    )
    result = candidate_aggregations(records)
    assert result["max_content_error_severity"] is None
    assert result["content_error_count"] == 0.0
    assert result["content_error_severity_sum"] == 0.0


def test_candidate_aggregations_pick_up_content_errors_only():
    item = {
        "source": "a",
        "target": "b",
        "metadata": {"system": "sys", "annotator": "ann"},
        "edits": [
            {
                "category": "deletion",
                "annotation": {
                    "deletion_type": {"val": "bad_deletion", "bad_deletion": "a lot"},
                    "grammar_error": "no",
                },
            },
            {
                "category": "deletion",
                "annotation": {
                    "deletion_type": {"val": "bad_deletion", "bad_deletion": "minor"},
                    "grammar_error": "no",
                },
            },
            {
                "category": "structure",
                "annotation": {
                    "structure_type": {"val": "voice"},
                    "impact": {"val": "bad", "bad": "a lot"},
                    "grammar_error": "no",
                },
            },
        ],
    }
    records, _ = parse_item_edits(item, 0)
    result = candidate_aggregations(records)
    assert result["max_content_error_severity"] == 3
    assert result["content_error_count"] == 2.0
    assert result["content_error_severity_sum"] == 4.0
