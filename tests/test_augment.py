"""Tests for train-split augmentation.

The property that matters most is the one v1 lacked: augmentation must not put a held-out
source sentence into train. Swapping turns the simplification into the source sentence, so
it can bridge a group that the splitter deliberately separated.
"""

import collections

import pytest
from datasets import DatasetDict

from data.augment import (
    MODES,
    assert_augmentation_safe,
    augment_splits,
    back_translate,
    generate_identical,
    generate_unrelated,
    swap,
    token_overlap,
)
from data.schema import build
from data.splits import LeakageError, group_key

REVERSE = lambda texts: [text[::-1] for text in texts]  # noqa: E731 - deterministic stub


def _dataset(rows):
    """Build a harmonised dataset from (original, simplification, source, label) tuples."""
    built = build(
        [
            {
                "item_id": str(index),
                "original": original,
                "simplification": simplification,
                "label_raw": float(label),
                "scale": "da100",
                "domain": "wiki",
                "license": "MIT",
                "source": source,
            }
            for index, (original, simplification, source, label) in enumerate(rows)
        ],
        "demo",
    )
    return built.remove_columns(["label"]).add_column("label", [float(r[3]) for r in rows])


#: Lexically disjoint sentences, so the unrelated generator has valid candidates.
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
    ("sailors navigated treacherous straits", "crews crossed narrows"),
]


def _train(n=6):
    return _dataset([(*_TOPICS[i % len(_TOPICS)], "original", 70.0) for i in range(n)])


# --- swap ----------------------------------------------------------------------------


def test_swap_mirrors_every_ordinary_pair():
    out = swap(_dataset([("a b c", "d e f", "original", 70.0)]))
    assert len(out) == 2
    assert out["original"][1] == "d e f"
    assert out["simplification"][1] == "a b c"


def test_swap_carries_the_label_over_unchanged():
    """The commutative property is the point: Meaning(A,B) = Meaning(B,A)."""
    out = swap(_dataset([("a b c", "d e f", "original", 42.5)]))
    assert out["label"][1] == pytest.approx(42.5)


def test_swap_tags_the_mirrored_rows():
    out = swap(_dataset([("a b c", "d e f", "original", 70.0)]))
    assert out["source"][1] == "swapped"


def test_swap_skips_identical_pairs():
    out = swap(_dataset([("a b c", "a b c", "identical", 100.0)]))
    assert len(out) == 1


def test_swap_does_not_recreate_a_pair_that_already_exists():
    out = swap(_dataset([("a b", "c d", "original", 70.0), ("c d", "a b", "original", 70.0)]))
    assert len(out) == 2


def test_swap_gives_mirrored_rows_a_distinct_item_id():
    out = swap(_dataset([("a b", "c d", "original", 70.0)]))
    assert len(set(out["item_id"])) == len(out)


def test_swap_drops_a_mirror_that_would_enter_a_held_out_group():
    """The v1 blind spot, measured on CSMD at the very first fold."""
    out = swap(_dataset([("a b", "held out sentence", "original", 70.0)]),
               forbidden_groups={group_key("held out sentence")})
    assert len(out) == 1


# --- back-translation ----------------------------------------------------------------


def test_back_translation_adds_one_variant_per_side():
    out = back_translate(_dataset([("abc", "def", "original", 70.0)]), REVERSE)
    assert len(out) == 3
    assert set(out["source"][1:]) == {"back_translated"}


def test_back_translation_keeps_the_label():
    out = back_translate(_dataset([("abc", "def", "original", 33.0)]), REVERSE)
    assert all(value == pytest.approx(33.0) for value in out["label"])


def test_a_paraphrase_identical_to_the_original_is_dropped():
    out = back_translate(_dataset([("aba", "def", "original", 70.0)]), lambda t: list(t))
    assert len(out) == 1


def test_back_translation_drops_a_variant_landing_in_a_held_out_group():
    out = back_translate(
        _dataset([("abc", "def", "original", 70.0)]),
        REVERSE,
        forbidden_groups={group_key("cba")},
    )
    # the bt_o variant would have source "cba"; only the bt_s variant survives
    assert len(out) == 2


def test_back_translation_respects_the_batch_size():
    seen = []

    def translate(texts):
        seen.append(len(texts))
        return [t[::-1] for t in texts]

    back_translate(_train(10), translate, batch_size=4)
    assert max(seen) <= 4


# --- generation ----------------------------------------------------------------------


def test_generated_identical_pairs_score_one_hundred():
    out = generate_identical(_train(10), ratio=0.5)
    generated = [row for row in out if row["system"] == "generated"]
    assert generated
    assert all(row["label"] == 100.0 for row in generated)
    assert all(row["original"] == row["simplification"] for row in generated)


def test_generated_identical_respects_the_ratio():
    out = generate_identical(_train(10), ratio=0.3)
    assert len(out) - 10 == 3


def test_a_zero_ratio_generates_nothing():
    assert len(generate_identical(_train(10), ratio=0.0)) == 10
    assert len(generate_unrelated(_train(10), ratio=0.0)) == 10


def test_generated_unrelated_pairs_score_zero():
    out = generate_unrelated(_train(10), ratio=0.3)
    generated = [row for row in out if row["system"] == "generated"]
    assert generated
    assert all(row["label"] == 0.0 for row in generated)


def test_generated_unrelated_pairs_do_not_share_content():
    out = generate_unrelated(_train(12), ratio=0.5, max_overlap=0.2)
    for row in out:
        if row["system"] == "generated":
            assert token_overlap(row["original"], row["simplification"]) <= 0.2


def test_generation_stops_instead_of_spinning_when_every_sentence_overlaps():
    """A corpus of near-duplicates has no valid unrelated pair; the loop must terminate."""
    dataset = _dataset([(f"the same words here {i}", f"x{i}", "original", 70.0) for i in range(8)])
    out = generate_unrelated(dataset, ratio=1.0, max_overlap=0.0)
    assert len(out) == len(dataset)


def test_generation_is_deterministic_for_a_given_seed():
    first = generate_unrelated(_train(12), ratio=0.4, seed=7)["item_id"]
    second = generate_unrelated(_train(12), ratio=0.4, seed=7)["item_id"]
    assert first == second


def test_token_overlap_is_containment_not_jaccard():
    """A short sentence inside a long one is not unrelated to it."""
    assert token_overlap("alpha beta", "alpha beta gamma delta epsilon") == pytest.approx(1.0)


def test_token_overlap_of_disjoint_sentences_is_zero():
    assert token_overlap("alpha beta", "gamma delta") == 0.0


# --- the safety check ----------------------------------------------------------------


def test_the_safety_check_catches_a_bridged_group():
    train = _dataset([("held out sentence", "x y", "original", 70.0)])
    other = _dataset([("held out sentence", "z w", "original", 70.0)])
    with pytest.raises(LeakageError, match="held-out source sentence"):
        assert_augmentation_safe(train, [other])


def test_the_safety_check_ignores_leakage_that_predates_augmentation():
    """Condition 'a' reproduces v1's leaking row-level split on purpose; the check must
    stay aimed at augmentation, the only thing this module controls."""
    baseline = _dataset([("held out sentence", "x y", "original", 70.0)])
    other = _dataset([("held out sentence", "z w", "original", 70.0)])
    assert_augmentation_safe(baseline, [other], baseline=baseline)


def test_the_safety_check_still_catches_a_group_augmentation_brought_in():
    baseline = _dataset([("clean sentence", "x y", "original", 70.0)])
    augmented = _dataset([("clean sentence", "x y", "original", 70.0), ("held out", "q", "swapped", 70.0)])
    other = _dataset([("held out", "z w", "original", 70.0)])
    with pytest.raises(LeakageError, match="augmentation added"):
        assert_augmentation_safe(augmented, [other], baseline=baseline)


def test_the_safety_check_passes_on_disjoint_splits():
    assert_augmentation_safe(_dataset([("a", "b", "original", 70.0)]), [_dataset([("c", "d", "original", 70.0)])])


# --- augment_splits ------------------------------------------------------------------


def _splits():
    return DatasetDict(
        {
            "train": _train(10),
            "dev": _dataset([("dev sentence alpha", "dev simpler", "original", 70.0)]),
            "test": _dataset([("test sentence omega", "test simpler", "original", 70.0)]),
            "sanity": _dataset([("sanity sentence", "sanity sentence", "identical", 100.0)]),
        }
    )


def test_there_are_exactly_two_conditions():
    assert MODES == ("none", "full")


def test_mode_none_changes_nothing():
    splits = _splits()
    out, counts = augment_splits(splits, "none")
    assert len(out["train"]) == len(splits["train"])
    assert counts["before"] == counts["after"]


def test_mode_full_applies_all_three_augmentations():
    out, counts = augment_splits(_splits(), "full", translate=REVERSE)
    sources = collections.Counter(out["train"]["source"])
    assert sources["swapped"] > 0
    assert sources["back_translated"] > 0
    assert sources["identical"] > 0
    assert sources["unrelated"] > 0
    assert counts["before"] < counts["after_generation"] < counts["after_swap"] < counts["after"]


def test_only_the_train_split_is_touched():
    splits = _splits()
    out, _ = augment_splits(splits, "full", translate=REVERSE)
    for name in ("dev", "test", "sanity"):
        assert out[name]["item_id"] == splits[name]["item_id"]


def test_the_augmented_train_split_never_leaks():
    out, _ = augment_splits(_splits(), "full", translate=REVERSE)
    assert_augmentation_safe(out["train"], [out["dev"], out["test"], out["sanity"]])


def test_full_without_a_translator_is_refused():
    with pytest.raises(ValueError, match="needs a translate callable"):
        augment_splits(_splits(), "full")


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown augmentation mode"):
        augment_splits(_splits(), "swap")


def test_item_ids_stay_unique_after_full_augmentation():
    out, _ = augment_splits(_splits(), "full", translate=REVERSE)
    ids = out["train"]["item_id"]
    assert len(ids) == len(set(ids))
