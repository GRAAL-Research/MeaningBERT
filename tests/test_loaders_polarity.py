"""Tests for the three polarity-only v3 loaders.

VitaminC, MoNLI and NaN-NLI share one shape: they annotate polarity and nothing
else, so they all declare ``scale = "none"`` with ``label_raw`` NaN. They are tested
together because what is worth testing in each is its *specific* risk, not the shape they
share. SICK is the exception and keeps its own file: it carries both targets and has a
join to defend.

Everything runs on in-memory fixtures. No network.
"""

from __future__ import annotations

import math

import pytest

from data.loaders import monli, nan_nli, vitaminc
from data.schema import build, validate

# --- VitaminC: the direction of the pair ---------------------------------------------

_VITAMINC = [
    {
        "unique_id": "abc_1",
        "label": "SUPPORTS",
        "claim": "Manchester had a population of more than 540,000 in 2017 .",
        "evidence": "Manchester is a major city with a population of 545,500 as of 2017 .",
        "revision_type": "real",
    },
    {
        "unique_id": "abc_2",
        "label": "REFUTES",
        "claim": "Manchester had a population of fewer than 100,000 in 2017 .",
        "evidence": "Manchester is a major city with a population of 545,500 as of 2017 .",
        "revision_type": "synthetic",
    },
]


def test_vitaminc_puts_the_evidence_first_and_the_claim_second():
    # The evidence is the premise and the claim is the hypothesis. Reversing them asks
    # whether the evidence follows from the claim, which is not what annotators answered.
    row = vitaminc._rows(_VITAMINC, "train")[0]
    assert row["original"].startswith("Manchester is a major city")
    assert row["simplification"].startswith("Manchester had a population of more than")


def test_vitaminc_declares_that_it_measures_no_meaning_preservation():
    # The dangerous default is 0.0, which reads as "no meaning preserved" instead of "not
    # measured" and would poison the magnitude head with half a million confident zeroes.
    rows = vitaminc._rows(_VITAMINC, "train")
    assert {row["scale"] for row in rows} == {"none"}
    assert all(math.isnan(row["label_raw"]) for row in rows)


def test_vitaminc_keeps_real_and_synthetic_revisions_distinguishable():
    rows = vitaminc._rows(_VITAMINC, "train")
    assert [row["system"] for row in rows] == ["vitaminc-real", "vitaminc-synthetic"]


def test_vitaminc_skips_a_row_whose_evidence_is_empty():
    rows = vitaminc._rows([dict(_VITAMINC[0], evidence="   ")], "train")
    assert rows == []


def test_vitaminc_stops_on_a_verdict_it_has_not_verified():
    with pytest.raises(vitaminc.LabelSpaceError, match="outside the verified label space"):
        vitaminc._rows([dict(_VITAMINC[0], label="MOSTLY TRUE")], "train")


def test_vitaminc_validates_against_the_contract():
    dataset = build(vitaminc._rows(_VITAMINC, "train"), corpus="vitaminc")
    validate(dataset)
    assert set(dataset["polarity_scheme"]) == {"fact3"}


# --- MoNLI: two classes, and the assumption that it stays that way -------------------

_MONLI = [
    {
        "sentence1": "There is a man not wearing a hat staring at people on a subway.",
        "sentence2": "There is a man not wearing a sunhat staring at people on a subway.",
        "gold_label": "entailment",
    },
    {
        "sentence1": "There is a man wearing a hat staring at people on a subway.",
        "sentence2": "There is a man wearing a sunhat staring at people on a subway.",
        "gold_label": "neutral",
    },
]


def test_monli_emits_only_two_of_the_three_nli_classes():
    rows = monli._rows(_MONLI, "train")
    assert {row["polarity_raw"] for row in rows} == {"entailment", "neutral"}
    assert {row["polarity_scheme"] for row in rows} == {"nli3"}


def test_a_contradiction_appearing_in_monli_stops_the_load():
    # MoNLI carrying no contradiction class is why the dissociation diagnostic reports it
    # separately instead of averaging it in. If the corpus gained one, that separate
    # treatment would no longer be justified, so the assumption is enforced and not assumed.
    with pytest.raises(monli.LabelSpaceError, match="no contradiction class"):
        monli._rows([dict(_MONLI[0], gold_label="contradiction")], "train")


def test_monli_validates_against_the_contract():
    dataset = build(monli._rows(_MONLI, "train"), corpus="monli")
    validate(dataset)
    assert set(dataset["polarity_scheme"]) == {"nli3"}
    assert set(dataset["scale"]) == {"none"}
    assert all(math.isnan(value) for value in dataset["label_raw"])


# --- NaN-NLI: a probe that must not leak into training -------------------------------

_NAN_NLI = [
    {
        "premise": "Not all people have had the opportunities you have had.",
        "hypothesis": "Some people have not had the opportunities you have had.",
        "label": "entailment",
    },
    {
        "premise": "Not all people have had the opportunities you have had.",
        "hypothesis": "Everyone has had the opportunities you have had.",
        "label": "contradiction",
    },
]


def test_nan_nli_refuses_to_propagate_its_published_train_split_name():
    # The corpus publishes one split, named 'train' by upload accident. Carrying that name
    # through would quietly let a 258-pair diagnostic probe into the training data, and the
    # probe would then measure nothing.
    rows = nan_nli._rows(_NAN_NLI)
    assert {row["split_hint"] for row in rows} == {""}


def test_nan_nli_carries_all_three_classes():
    rows = nan_nli._rows(_NAN_NLI)
    assert {row["polarity_raw"] for row in rows} == {"entailment", "contradiction"}


def test_nan_nli_stops_on_a_label_it_has_not_verified():
    with pytest.raises(nan_nli.LabelSpaceError, match="outside the verified label space"):
        nan_nli._rows([dict(_NAN_NLI[0], label="unknown")])


def test_nan_nli_validates_against_the_contract():
    dataset = build(nan_nli._rows(_NAN_NLI), corpus="nan_nli")
    validate(dataset)
    assert set(dataset["polarity_scheme"]) == {"nli3"}
    assert set(dataset["scale"]) == {"none"}
    assert all(math.isnan(value) for value in dataset["label_raw"])
