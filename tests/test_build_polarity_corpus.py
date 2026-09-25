"""Tests for the polarity corpus builder (``src/data/build_polarity_corpus.py``).

This is the step that reconciles label spaces written by different people for different
papers, which is the single place where a whole corpus can be silently mislabelled. It is
also where the probes are kept out of training, and a probe that has been trained on
measures nothing while still reporting a number.
"""

from __future__ import annotations

import pytest
from datasets import Dataset

from data.build_polarity_corpus import (
    POLARITY_MAP,
    PROBE_CORPORA,
    TRAINING_CORPORA,
    BuildError,
    assert_no_pair_leakage,
    cap_per_class,
    unify,
)
from data.schema import POLARITY_CLASSES, build


def _dataset(rows, corpus="demo"):
    return build(rows, corpus)


def _row(index, polarity, scheme="nli3", **overrides):
    row = {
        "item_id": str(index),
        "original": f"premise {index}",
        "simplification": f"hypothesis {index}",
        "label_raw": float("nan"),
        "scale": "none",
        "domain": "wiki",
        "license": "MIT",
        "polarity_raw": polarity,
        "polarity_scheme": scheme,
    }
    row.update(overrides)
    return row


# --- unify: reconciling the label spaces ---------------------------------------------


def test_the_nli_scheme_maps_onto_itself():
    unified = unify(_dataset([_row(0, "entailment"), _row(1, "contradiction"), _row(2, "neutral")]))
    assert unified["polarity"] == [
        float(POLARITY_CLASSES["entailment"]),
        float(POLARITY_CLASSES["contradiction"]),
        float(POLARITY_CLASSES["neutral"]),
    ]


def test_a_refuted_claim_becomes_a_contradiction_and_a_supported_one_an_entailment():
    # The fact-checking relation is the inference relation under another name: a claim
    # refuted by its evidence is contradicted by it.
    rows = [_row(0, "REFUTES", "fact3"), _row(1, "SUPPORTS", "fact3"), _row(2, "NOT ENOUGH INFO", "fact3")]
    unified = unify(_dataset(rows))
    assert unified["polarity"] == [
        float(POLARITY_CLASSES["contradiction"]),
        float(POLARITY_CLASSES["entailment"]),
        float(POLARITY_CLASSES["neutral"]),
    ]


def test_the_two_schemes_agree_on_what_a_contradiction_is():
    # The property the whole merge rests on: after unification, a REFUTES row and a
    # contradiction row are indistinguishable to the head.
    fact = unify(_dataset([_row(0, "REFUTES", "fact3")]))
    nli = unify(_dataset([_row(0, "contradiction")]))
    assert fact["polarity"] == nli["polarity"]


def test_an_unmappable_raw_value_stops_the_build_instead_of_dropping_the_row():
    dataset = _dataset([_row(0, "entailment")])
    dataset = dataset.remove_columns(["polarity_raw"]).add_column("polarity_raw", ["MOSTLY TRUE"])
    with pytest.raises(BuildError, match="no unified class"):
        unify(dataset)


def test_an_unknown_scheme_stops_the_build():
    dataset = _dataset([_row(0, "entailment")])
    dataset = dataset.remove_columns(["polarity_scheme"]).add_column("polarity_scheme", ["vibes3"])
    with pytest.raises(BuildError, match="no unified mapping"):
        unify(dataset)


def test_every_declared_scheme_is_total_over_its_own_vocabulary():
    # A mapping that covers only part of a scheme would fail on real data long after the
    # build looked fine on a fixture.
    from data.schema import POLARITY_SCHEMES

    for scheme, mapping in POLARITY_MAP.items():
        assert set(mapping) == set(POLARITY_SCHEMES[scheme])
        assert set(mapping.values()) <= set(POLARITY_CLASSES)


# --- the cap -------------------------------------------------------------------------


def test_the_cap_applies_per_class_and_preserves_the_smaller_classes_whole():
    rows = [_row(i, "entailment") for i in range(10)] + [_row(100 + i, "contradiction") for i in range(3)]
    capped = cap_per_class(unify(_dataset(rows)), cap=4, seed=42)
    counts = {value: capped["polarity"].count(value) for value in set(capped["polarity"])}
    assert counts[float(POLARITY_CLASSES["entailment"])] == 4
    assert counts[float(POLARITY_CLASSES["contradiction"])] == 3


def test_capping_is_reproducible_across_runs():
    rows = [_row(i, "entailment") for i in range(20)]
    unified = unify(_dataset(rows))
    assert cap_per_class(unified, 5, 42)["item_id"] == cap_per_class(unified, 5, 42)["item_id"]


def test_a_different_seed_draws_a_different_sample():
    rows = [_row(i, "entailment") for i in range(40)]
    unified = unify(_dataset(rows))
    assert cap_per_class(unified, 5, 1)["item_id"] != cap_per_class(unified, 5, 2)["item_id"]


def test_no_cap_keeps_everything():
    unified = unify(_dataset([_row(i, "entailment") for i in range(7)]))
    assert len(cap_per_class(unified, None, 42)) == 7


# --- leakage -------------------------------------------------------------------------


def _split(pairs):
    return Dataset.from_dict(
        {"original": [left for left, _ in pairs], "simplification": [right for _, right in pairs]}
    )


def test_disjoint_splits_pass():
    assert_no_pair_leakage(
        {"train": _split([("a", "b")]), "dev": _split([("c", "d")]), "test": _split([("e", "f")])}
    )


def test_a_pair_shared_between_train_and_test_is_refused():
    # The corpora publish their own splits and those are trusted, but they were built
    # independently of one another, so nothing guarantees the union is clean.
    with pytest.raises(BuildError, match="train and test"):
        assert_no_pair_leakage(
            {"train": _split([("a", "b")]), "dev": _split([("c", "d")]), "test": _split([("a", "b")])}
        )


def test_leakage_is_detected_through_case_and_whitespace():
    with pytest.raises(BuildError, match="train and dev"):
        assert_no_pair_leakage(
            {"train": _split([("A dog barks", "no dog barks")]),
             "dev": _split([("  a dog barks ", "NO DOG BARKS")]),
             "test": _split([("e", "f")])}
        )


# --- what trains and what does not ---------------------------------------------------


def test_the_probes_are_not_among_the_training_corpora():
    # MoNLI and NaN-NLI are two of the three suites of the dissociation diagnostic. A probe
    # that has been trained on still reports a number, which is what makes it dangerous.
    assert set(TRAINING_CORPORA) & set(PROBE_CORPORA) == set()
    assert set(PROBE_CORPORA) == {"monli", "nan_nli"}
