"""Tests for the polarity corpus builder (``src/data/build_polarity_corpus.py``).

This is the step that reconciles label spaces written by different people for different
papers, which is the single place where a whole corpus can be silently mislabelled. It is
also where the probes are kept out of training, and a probe that has been trained on
measures nothing while still reporting a number.
"""

from __future__ import annotations

import collections

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


# --- augmentation: every added row's polarity is derived, never copied ---------------

from data.build_polarity_corpus import (  # noqa: E402
    AUGMENTATION_TARGETS,
    augment_and_verify,
    augment_polarity,
)


def _train(pairs):
    """A harmonised training split from (left, right, polarity) triples."""
    rows = [
        _row(index, polarity, original=left, simplification=right)
        for index, (left, right, polarity) in enumerate(pairs)
    ]
    return unify(_dataset(rows))


#: Lexically disjoint pairs, so the unrelated generator has valid candidates to draw from.
_TOPICS = [
    ("volcanoes erupt molten basalt", "lava flows downhill"),
    ("bankers approved mortgage lending", "loans were granted"),
    ("penguins huddle against blizzards", "birds keep warm"),
    ("compilers optimise register allocation", "software runs faster"),
    ("surgeons transplanted kidney tissue", "doctors moved an organ"),
    ("archaeologists dated pottery shards", "experts aged ceramics"),
    ("violinists rehearsed concerto passages", "musicians practised"),
    ("farmers irrigate drought stricken fields", "crops receive water"),
    ("astronomers catalogued distant quasars", "scientists listed galaxies"),
    ("legislators amended zoning statutes", "lawmakers changed rules"),
    ("chemists synthesised polymer chains", "researchers made plastic"),
    ("engineers reinforced bridge abutments", "builders strengthened supports"),
]


def _mixed_train():
    cycle = ["contradiction", "entailment", "neutral"]
    return _train([(left, right, cycle[i % 3]) for i, (left, right) in enumerate(_TOPICS)])


def _rows_of(dataset):
    return [dict(zip(dataset.column_names, values)) for values in zip(*(dataset[c] for c in dataset.column_names))]


def test_only_contradictions_survive_the_mirror():
    # Contradiction is symmetric; entailment and neutral are not. A mirrored entailment
    # would arrive with no polarity AND no magnitude label, so it carries no target at all.
    augmented, census = augment_polarity(_mixed_train(), forbidden_groups=set(), per_class=50, seed=42)
    mirrored = [row for row in _rows_of(augmented) if row["source"] == "swapped"]
    assert mirrored
    assert {row["polarity_raw"] for row in mirrored} == {"contradiction"}
    assert census["swapped"] == len(mirrored)


def test_the_generated_unrelated_pairs_are_neutral_not_contradictions():
    # The distinction the whole signed scale rests on: unrelated scores 0, not -100.
    augmented, _ = augment_polarity(_mixed_train(), forbidden_groups=set(), per_class=50, seed=42)
    generated = [row for row in _rows_of(augmented) if row["source"] == "unrelated"]
    assert generated
    assert {row["polarity_raw"] for row in generated} == {"neutral"}


def test_the_generated_identical_pairs_are_entailments():
    augmented, _ = augment_polarity(_mixed_train(), forbidden_groups=set(), per_class=50, seed=42)
    generated = [row for row in _rows_of(augmented) if row["source"] == "identical"]
    assert generated
    assert {row["polarity_raw"] for row in generated} == {"entailment"}
    assert {row["original"] == row["simplification"] for row in generated} == {True}


def test_each_augmentation_feeds_a_different_class():
    # The mechanical reason all three are included: mirroring alone grows only the
    # contradiction class, and a head that over-predicts the one class whose sign flips
    # the score is exactly the failure to avoid.
    assert set(AUGMENTATION_TARGETS.values()) == {"entailment", "neutral", "contradiction"}
    assert len(AUGMENTATION_TARGETS) == 3


def test_the_unified_polarity_column_matches_the_derived_name():
    # polarity_raw is provenance; polarity is what the head trains on. The two must agree,
    # or the report and the target describe different corpora.
    augmented, _ = augment_polarity(_mixed_train(), forbidden_groups=set(), per_class=50, seed=42)
    for row in _rows_of(augmented):
        if row["source"] in AUGMENTATION_TARGETS:
            assert row["polarity"] == float(POLARITY_CLASSES[row["polarity_raw"]])


def test_the_quota_caps_each_augmentation():
    augmented, census = augment_polarity(_mixed_train(), forbidden_groups=set(), per_class=2, seed=42)
    assert all(count <= 2 for count in census.values())
    for source in AUGMENTATION_TARGETS:
        assert len([row for row in _rows_of(augmented) if row["source"] == source]) <= 2


def test_augmentation_never_removes_an_original_row():
    train = _mixed_train()
    augmented, _ = augment_polarity(train, forbidden_groups=set(), per_class=50, seed=42)
    assert len([row for row in _rows_of(augmented) if row["source"] == "original"]) == len(train)
    assert len(augmented) > len(train)


def test_a_mirror_landing_in_a_held_out_group_is_dropped():
    # Swapping moves the simplification into the source position, so it can bridge a group
    # the splitter deliberately separated. This leak is invisible to a whole-pair check.
    from data.splits import group_key

    train = _train([(left, right, "contradiction") for left, right in _TOPICS])
    forbidden = {group_key(right) for _, right in _TOPICS[:6]}
    augmented, census = augment_polarity(train, forbidden_groups=forbidden, per_class=50, seed=42)
    mirrored = [row for row in _rows_of(augmented) if row["source"] == "swapped"]
    assert census["swapped"] == len(mirrored) == 6
    assert not {group_key(row["original"]) for row in mirrored} & forbidden


def test_the_generators_are_asked_for_more_than_the_quota_then_cut_back():
    # The quota guard has to be load-bearing, not decorative: the unrelated generator
    # rejects overlapping draws and the identical one skips repeats, so asking for exactly
    # the quota returns less than it. Overshooting then cutting back is what keeps the
    # three classes growing by the same amount.
    augmented, census = augment_polarity(_mixed_train(), forbidden_groups=set(), per_class=6, seed=42)
    assert census["unrelated"] == 6
    assert census["identical"] == 6
    assert len([row for row in _rows_of(augmented) if row["source"] == "unrelated"]) == 6


def _eval_split(pairs):
    """A minimal dev/test split: augment_and_verify only reads originals and pairs."""
    return Dataset.from_dict(
        {"original": [left for left, _ in pairs], "simplification": [right for _, right in pairs]}
    )


def test_a_mirror_into_a_held_out_group_is_stopped_before_the_second_check():
    # Defence in depth, and the order matters: forbidden_groups stops the mirror at the
    # swap itself, so the second check never has to see it.
    left, right = _TOPICS[0]
    splits = {
        "train": _train([(left, right, "contradiction")]),
        "dev": _eval_split([("unrelated left", "unrelated right")]),
        "test": _eval_split([(right, left)]),
    }
    census = augment_and_verify(splits, per_class=50, seed=42)
    assert census["swapped"] == 0


def test_a_generated_pair_that_lands_in_the_test_split_is_caught():
    # The leak forbidden_groups does NOT cover: it guards the mirror only, and the
    # generators build brand new pairs from the training sentences. Here the test split
    # already holds the identical pair (A, A) that generate_identical is about to invent,
    # and the only thing standing between that and a corrupted test set is the check that
    # runs AFTER augmentation.
    sentence = _TOPICS[0][0]
    splits = {
        "train": _train([(left, right, "contradiction") for left, right in _TOPICS]),
        "dev": _eval_split([("a held out sentence", "its candidate")]),
        "test": _eval_split([(sentence, sentence)]),
    }
    with pytest.raises(BuildError, match="train and test"):
        augment_and_verify(splits, per_class=50, seed=42)


def test_a_clean_augmentation_passes_the_second_check_and_reports_its_census():
    splits = {
        "train": _mixed_train(),
        "dev": _eval_split([("a held out sentence", "its candidate")]),
        "test": _eval_split([("another held out one", "another candidate")]),
    }
    census = augment_and_verify(splits, per_class=4, seed=42)
    assert set(census) == set(AUGMENTATION_TARGETS)
    assert len(splits["train"]) > 12


def test_a_refuted_vitaminc_pair_mirrors_just_like_a_sick_contradiction():
    # The bug this pins down cost 50 000 rows: the symmetry rule was written against
    # polarity_raw, so it matched SICK's "contradiction" and silently missed VitaminC's
    # "REFUTES", which is the same relation under a fact-checking name. 99.4 % of the
    # mirrors vanished and the contradiction class stopped growing.
    rows = [
        _row(index, "REFUTES", scheme="fact3", original=left, simplification=right)
        for index, (left, right) in enumerate(_TOPICS)
    ]
    train = unify(_dataset(rows))
    augmented, census = augment_polarity(train, forbidden_groups=set(), per_class=50, seed=42)
    assert census["swapped"] == len(_TOPICS)
    mirrored = [row for row in _rows_of(augmented) if row["source"] == "swapped"]
    assert {row["polarity"] for row in mirrored} == {float(POLARITY_CLASSES["contradiction"])}


def test_the_two_schemes_mirror_at_the_same_rate():
    # The property that makes the merge legitimate: after unification, a fact3 corpus and
    # an nli3 corpus must behave identically under augmentation.
    def census_for(raw, scheme):
        rows = [
            _row(i, raw, scheme=scheme, original=left, simplification=right)
            for i, (left, right) in enumerate(_TOPICS)
        ]
        return augment_polarity(unify(_dataset(rows)), set(), per_class=50, seed=42)[1]["swapped"]

    assert census_for("REFUTES", "fact3") == census_for("contradiction", "nli3")


def test_a_supported_vitaminc_pair_still_loses_its_polarity_when_mirrored():
    # The rule must not become "everything is symmetric now": SUPPORTS maps to entailment,
    # and entailment does not survive the swap any more than it did before.
    rows = [
        _row(index, "SUPPORTS", scheme="fact3", original=left, simplification=right)
        for index, (left, right) in enumerate(_TOPICS)
    ]
    augmented, census = augment_polarity(unify(_dataset(rows)), set(), per_class=50, seed=42)
    assert census["swapped"] == 0


# --- stratification -------------------------------------------------------------------

from data.build_polarity_corpus import (  # noqa: E402
    assert_no_group_leakage,
    drop_leaked_groups,
    stratify,
)


def _mixed_corpus_rows():
    """A big corpus and a small one, both carrying all three classes."""
    rows = []
    cycle = ["entailment", "neutral", "contradiction"]
    for index in range(90):
        rows.append(_row(index, cycle[index % 3], original=f"big premise {index}"))
    big = unify(_dataset(rows, corpus="vitaminc"))
    rows = []
    for index in range(9):
        rows.append(_row(index, cycle[index % 3], original=f"small premise {index}"))
    small = unify(_dataset(rows, corpus="sick"))
    columns = big.column_names
    from datasets import concatenate_datasets

    return concatenate_datasets([big.select_columns(columns), small.select_columns(columns)])


def test_stratify_balances_the_three_classes():
    # The corpora's own evaluation splits are 48 % entailment against 16 % neutral, and a
    # head that never predicts neutral still scores respectably on such a set.
    drawn = stratify(_mixed_corpus_rows(), per_class=6, seed=42)
    counts = collections.Counter(drawn["polarity"])
    assert set(counts.values()) == {6}
    assert len(counts) == 3


def test_stratify_takes_a_small_corpus_whole_rather_than_to_its_natural_share():
    # SICK is 0.8 % of the merged dev split because VitaminC is fifty times its size, and
    # it is the bridge corpus. Sampling it down to its proportion is the failure here.
    drawn = stratify(_mixed_corpus_rows(), per_class=6, seed=42)
    by_corpus = collections.Counter(drawn["corpus"])
    assert by_corpus["sick"] == 9
    assert by_corpus["vitaminc"] == 9


def test_the_surplus_a_small_corpus_cannot_use_spills_to_the_large_one():
    # Water-filling: the share sick cannot fill is redistributed, not lost, so the class
    # quota is still met exactly.
    drawn = stratify(_mixed_corpus_rows(), per_class=10, seed=42)
    counts = collections.Counter(drawn["polarity"])
    assert set(counts.values()) == {10}
    assert collections.Counter(drawn["corpus"])["sick"] == 9


def test_stratify_is_reproducible():
    data = _mixed_corpus_rows()
    assert stratify(data, 6, 42)["item_id"] == stratify(data, 6, 42)["item_id"]


def test_stratify_never_invents_rows_a_class_does_not_have():
    rows = [_row(i, "entailment") for i in range(4)] + [_row(100 + i, "neutral") for i in range(2)]
    drawn = stratify(unify(_dataset(rows)), per_class=10, seed=42)
    counts = collections.Counter(drawn["polarity"])
    assert counts[float(POLARITY_CLASSES["entailment"])] == 4
    assert counts[float(POLARITY_CLASSES["neutral"])] == 2


# --- source-sentence leakage ----------------------------------------------------------


def test_a_training_row_sharing_a_source_sentence_with_the_test_split_is_dropped():
    # VitaminC reuses the same Wikipedia evidence across many claims and splits by case,
    # so 5.6 % of test source sentences were also in training. Not pair leakage, which the
    # pair check already refuses: the source-sentence leakage v2 was rebuilt to remove.
    shared = _TOPICS[0][0]
    splits = {
        "train": _train([(shared, "a different claim", "contradiction"), _TOPICS[1] + ("neutral",)]),
        "dev": _eval_split([("a held out premise", "its hypothesis")]),
        "test": _eval_split([(shared, "yet another claim")]),
    }
    dropped = drop_leaked_groups(splits)
    assert dropped == 1
    assert shared not in splits["train"]["original"]


def test_dropping_leaked_groups_keeps_the_rows_that_do_not_leak():
    splits = {
        "train": _train([(left, right, "contradiction") for left, right in _TOPICS]),
        "dev": _eval_split([("a held out premise", "its hypothesis")]),
        "test": _eval_split([("another held out premise", "its hypothesis")]),
    }
    assert drop_leaked_groups(splits) == 0
    assert len(splits["train"]) == len(_TOPICS)


def test_group_leakage_is_refused_outright_not_only_dropped():
    shared = _TOPICS[0][0]
    splits = {
        "train": _train([(shared, "a claim", "contradiction")]),
        "dev": _eval_split([("a held out premise", "its hypothesis")]),
        "test": _eval_split([(shared, "another claim")]),
    }
    with pytest.raises(BuildError, match="source-sentence leakage"):
        assert_no_group_leakage(splits)


def test_a_clean_corpus_passes_the_group_check():
    splits = {
        "train": _train([(left, right, "neutral") for left, right in _TOPICS]),
        "dev": _eval_split([("a held out premise", "its hypothesis")]),
        "test": _eval_split([("another held out premise", "its hypothesis")]),
    }
    assert_no_group_leakage(splits)


def test_dev_is_cleaned_against_test_too():
    # A development split sharing source sentences with the test split makes model
    # selection quietly optimistic about the number that gets reported. Dev is the split
    # that can afford to lose rows: it selects a model and never reports a number.
    shared = _TOPICS[0][0]
    splits = {
        "train": _train([_TOPICS[1] + ("neutral",)]),
        "dev": _train([(shared, "a hypothesis", "contradiction"), _TOPICS[2] + ("entailment",)]),
        "test": _eval_split([(shared, "another hypothesis")]),
    }
    assert drop_leaked_groups(splits, "dev", ("test",)) == 1
    assert shared not in splits["dev"]["original"]
    assert len(splits["dev"]) == 1


def test_the_group_check_covers_the_dev_against_test_wall():
    shared = _TOPICS[0][0]
    splits = {
        "train": _train([_TOPICS[1] + ("neutral",)]),
        "dev": _train([(shared, "a hypothesis", "contradiction")]),
        "test": _eval_split([(shared, "another hypothesis")]),
    }
    with pytest.raises(BuildError, match="both dev and test"):
        assert_no_group_leakage(splits)


# --- the generated sanity suite -------------------------------------------------------

from data.build_polarity_corpus import build_sanity_suite  # noqa: E402


def _suite(per_source=3, seed=42):
    splits = {"test": _train([(left, right, "contradiction") for left, right in _TOPICS])}
    return build_sanity_suite(splits, per_source=per_source, seed=seed)


def test_the_sanity_suite_carries_its_three_generated_families():
    got = collections.Counter(_suite()["source"])
    assert set(got) == {"identical", "unrelated", "swapped"}


def test_an_identical_pair_must_be_an_entailment():
    rows = [row for row in _rows_of(_suite()) if row["source"] == "identical"]
    assert rows
    assert {row["polarity"] for row in rows} == {float(POLARITY_CLASSES["entailment"])}
    assert all(row["original"] == row["simplification"] for row in rows)


def test_an_unrelated_pair_must_be_neutral_and_never_a_contradiction():
    # The check the whole signed scale lives on: a model reading "unrelated" as "opposed"
    # answers -100 where the truth is 0.
    rows = [row for row in _rows_of(_suite()) if row["source"] == "unrelated"]
    assert rows
    assert {row["polarity"] for row in rows} == {float(POLARITY_CLASSES["neutral"])}


def test_a_mirrored_contradiction_must_still_be_a_contradiction():
    rows = [row for row in _rows_of(_suite()) if row["source"] == "swapped"]
    assert rows
    assert {row["polarity"] for row in rows} == {float(POLARITY_CLASSES["contradiction"])}


def test_the_suite_is_built_from_test_sentences_only():
    # It inherits the wall that training was already cleaned against, instead of needing
    # one of its own.
    splits = {"test": _train([(left, right, "contradiction") for left, right in _TOPICS])}
    known = set(splits["test"]["original"]) | set(splits["test"]["simplification"])
    for row in _rows_of(build_sanity_suite(splits, per_source=5, seed=42)):
        assert row["original"] in known
        assert row["simplification"] in known


def test_the_same_seed_gives_both_conditions_a_byte_identical_suite():
    # _none and _full must be scored on the same suite, or the comparison means nothing.
    assert _suite()["item_id"] == _suite()["item_id"]


def test_the_suite_respects_its_per_source_quota():
    got = collections.Counter(_suite(per_source=2)["source"])
    assert all(count <= 2 for count in got.values())
