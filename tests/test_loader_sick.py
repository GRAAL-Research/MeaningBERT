"""Tests for the SICK loader (``src/data/loaders/sick.py``).

Runs entirely on in-memory fixtures: no network. SICK is the bridge between the two heads,
the only corpus carrying relatedness and inference on the same pairs, so the tests here are
about the join staying honest and the label encoding staying the one that was verified.
"""

from __future__ import annotations

import math

import pytest

from data.loaders.sick import (
    JoinError,
    _join,
    _relatedness_index,
    check_join_loss,
)
from data.schema import build, validate

# Two real SICK pairs, one neutral and one contradiction, with their real relatedness.
_ENTAILMENT = [
    {
        "text1": "A group of kids is playing in a yard and an old man is standing in the background",
        "text2": "A group of boys in a yard is playing and a man is standing in the background",
        "label": 1,
    },
    {"text1": "Two dogs are wrestling and hugging", "text2": "There is no dog wrestling and hugging", "label": 2},
    {"text1": "A man is playing a guitar", "text2": "A man is playing an instrument", "label": 0},
]
_RELATEDNESS = [
    {
        "sentence1": "A group of kids is playing in a yard and an old man is standing in the background",
        "sentence2": "A group of boys in a yard is playing and a man is standing in the background",
        "score": 4.5,
    },
    {"sentence1": "Two dogs are wrestling and hugging", "sentence2": "There is no dog wrestling and hugging", "score": 3.5},
    {"sentence1": "A man is playing a guitar", "sentence2": "A man is playing an instrument", "score": 4.7},
]


def _rows(entailment=None, relatedness=None, split="test"):
    return _join(entailment or _ENTAILMENT, split, _relatedness_index(relatedness or _RELATEDNESS))


# --- the join ------------------------------------------------------------------------


def test_the_join_attaches_each_pair_its_own_relatedness_score():
    rows, _ = _rows()
    assert [row["label_raw"] for row in rows] == [4.5, 3.5, 4.7]


def test_the_join_survives_whitespace_differences_between_the_two_halves():
    padded = [dict(_RELATEDNESS[0], sentence1=f"  {_RELATEDNESS[0]['sentence1']} ")]
    rows, unjoined = _rows(entailment=_ENTAILMENT[:1], relatedness=padded)
    assert unjoined == 0
    assert rows[0]["label_raw"] == 4.5


def test_the_join_does_not_merge_a_pair_with_its_mirror():
    # SICK publishes A/B and B/A as distinct rows with distinct labels. Normalising the
    # order would silently collapse rows that disagree with each other.
    mirrored = [{"text1": _ENTAILMENT[0]["text2"], "text2": _ENTAILMENT[0]["text1"], "label": 1}]
    rows, unjoined = _rows(entailment=mirrored)
    assert rows == []
    assert unjoined == 1


def test_a_pair_with_no_relatedness_score_is_dropped_and_counted():
    rows, unjoined = _rows(relatedness=_RELATEDNESS[:1])
    assert len(rows) == 1
    assert unjoined == 2


def test_an_unknown_label_code_stops_the_load_instead_of_skipping_the_row():
    # A new code means the upstream encoding moved, which makes every other row's polarity
    # suspect. Skipping the row would hide that.
    with pytest.raises(JoinError, match="outside the verified encoding"):
        _rows(entailment=[dict(_ENTAILMENT[0], label=3)])


# --- the label encoding, verified in the data ----------------------------------------


def test_the_verified_encoding_is_zero_entailment_one_neutral_two_contradiction():
    rows, _ = _rows()
    assert [row["polarity_raw"] for row in rows] == ["neutral", "contradiction", "entailment"]


def test_the_contradiction_pair_is_the_one_carrying_the_negation():
    rows, _ = _rows()
    contradiction = next(row for row in rows if row["polarity_raw"] == "contradiction")
    assert "no dog" in contradiction["simplification"]


# --- contract ------------------------------------------------------------------------


def test_sick_declares_both_targets_and_validates():
    rows, _ = _rows()
    dataset = build(rows, corpus="sick")
    validate(dataset)
    assert set(dataset["scale"]) == {"likert5"}
    assert set(dataset["polarity_scheme"]) == {"nli3"}
    assert all(math.isnan(value) for value in dataset["label"])
    assert all(math.isnan(value) for value in dataset["polarity"])


def test_the_validation_split_becomes_the_contract_dev_hint():
    rows, _ = _rows(split="validation")
    assert {row["split_hint"] for row in rows} == {"dev"}


# --- the guard on the bridge ---------------------------------------------------------


def test_a_clean_join_passes_the_guard():
    check_join_loss(0, 9840)


def test_losing_a_handful_of_pairs_is_tolerated():
    check_join_loss(50, 9840)


def test_losing_a_large_share_of_the_bridge_is_refused():
    # A smaller SICK is not a smaller inconvenience: it is the only corpus carrying both
    # targets, so a silent shrink biases the calibration of the whole composition.
    with pytest.raises(JoinError, match="no longer line up"):
        check_join_loss(2000, 9840)
