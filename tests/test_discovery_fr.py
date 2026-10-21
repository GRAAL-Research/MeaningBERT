"""Tests for the v4 discovery (``src/diagnostics/discovery_fr.py``).

The discovery decides whether the v4 exists, from one AUC gap on DACCORD. Every way that
number can come out wrong while looking plausible is pinned here: a contradiction logit
read at the wrong index, a corpus whose labels are mapped onto the wrong relation, SICK-fr
pairs the v3 head was trained on leaking into the test, scores scattered back onto the
wrong pairs, and a bootstrap that does not pair the two heads.
"""

from __future__ import annotations

import numpy as np
import pytest

from diagnostics.discovery_fr import (
    bootstrap_auc_gap,
    contradiction_index,
    delong,
    holm,
    inputs_for,
    mcnemar,
    normalize,
    report,
    slice_report,
    tables,
    welch,
)


def test_contradiction_index_reads_the_config_whatever_the_order() -> None:
    # joeddav/xlm-roberta-large-xnli puts contradiction first, the v3 heads last.
    assert contradiction_index({0: "contradiction", 1: "neutral", 2: "entailment"}) == 0
    assert contradiction_index({"0": "entailment", "1": "neutral", "2": "contradiction"}) == 2
    assert contradiction_index({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}) == 2


def test_contradiction_index_refuses_a_config_without_one() -> None:
    with pytest.raises(ValueError):
        contradiction_index({0: "LABEL_0", 1: "LABEL_1"})


def test_daccord_label_one_is_contradiction_and_zero_is_compatible() -> None:
    rows = [
        {"premise": "A", "hypothesis": "B", "label": "1", "genre": "covid"},
        {"premise": "C", "hypothesis": "D", "label": "0", "genre": "covid"},
    ]
    pairs = normalize("daccord", rows)
    assert [p["label"] for p in pairs] == ["contradiction", "compatible"]
    # DACCORD has no English original: it must never feed the ``orig`` input.
    assert all(p["en_a"] == "" for p in pairs)


def test_sick_fr_drops_pairs_the_v3_head_has_seen() -> None:
    rows = [
        {
            "sentence_A": "Un chien court",
            "sentence_B": "Un chien ne court pas",
            "entailment_label": "CONTRADICTION",
            "original_SICK_sentence_A": "A dog runs",
            "original_SICK_sentence_B": "A dog does not run",
        },
        {
            "sentence_A": "Un chat dort",
            "sentence_B": "Un animal dort",
            "entailment_label": "ENTAILMENT",
            "original_SICK_sentence_A": "A cat sleeps",
            "original_SICK_sentence_B": "An animal sleeps",
        },
    ]
    pairs = normalize("sick_fr", rows, seen=frozenset({("A dog runs", "A dog does not run")}))
    assert [p["en_a"] for p in pairs] == ["A cat sleeps"]
    assert pairs[0]["label"] == "entailment"


def test_fracas_maps_answers_and_drops_undefined_ones() -> None:
    base = {"premises": "P", "hypothesis": "H", "premises_original": "P", "hypothesis_original": "H"}
    rows = [{**base, "answer": "yes"}, {**base, "answer": "no"}, {**base, "answer": "unknown"}, {**base, "answer": ""}]
    assert [p["label"] for p in normalize("fracas_fr", rows)] == ["entailment", "contradiction", "neutral"]


def test_breaking_keeps_the_lexical_category() -> None:
    rows = [
        {
            "sentence1_fr": "Il fait chaud.",
            "sentence2_fr": "Il fait froid.",
            "gold_label": "contradiction",
            "category": "antonyms",
            "sentence1_en_orig": "It is hot.",
            "sentence2_en_orig": "It is cold.",
        }
    ]
    assert normalize("breaking_fr", rows)[0]["category"] == "antonyms"


def test_unknown_corpus_is_refused() -> None:
    with pytest.raises(ValueError):
        normalize("nope", [{}])


def test_orig_input_skips_pairs_without_an_english_original() -> None:
    pairs = [
        {"fr_a": "a", "fr_b": "b", "en_a": "", "en_b": ""},
        {"fr_a": "c", "fr_b": "d", "en_a": "C", "en_b": "D"},
    ]
    index, left, right = inputs_for(pairs, "orig", {})
    assert (index, left, right) == ([1], ["C"], ["D"])
    index, left, _ = inputs_for(pairs, "mt", {"a": "A*", "b": "B*", "c": "C*", "d": "D*"})
    assert (index, left) == ([0, 1], ["A*", "C*"])


def test_slice_report_below_zero_is_p_contra_above_half() -> None:
    p = np.array([0.9, 0.6, 0.1, 0.4])
    labels = np.array(["contradiction", "contradiction", "entailment", "entailment"])
    out = slice_report(p, labels, "entailment")
    assert out["auc"] == 1.0
    assert out["below_zero_contradiction"] == 1.0
    assert out["below_zero_entailment"] == 0.0


def test_bootstrap_gap_is_zero_for_identical_heads_and_positive_for_a_better_one() -> None:
    rng = np.random.default_rng(0)
    pos, neg = rng.uniform(0.4, 1.0, 200), rng.uniform(0.0, 0.6, 200)
    same = bootstrap_auc_gap(pos, neg, pos, neg, rounds=200)
    assert same["gap"] == 0.0 and same["low"] == 0.0 and same["high"] == 0.0
    better = bootstrap_auc_gap(pos + 1.0, neg, pos, neg, rounds=200)
    assert better["gap"] > 0 and better["low"] > 0


def _pairs() -> list[dict]:
    return [
        {"corpus": "daccord", "label": "compatible", "category": "", "en_a": "", "en_b": ""},
        {"corpus": "daccord", "label": "compatible", "category": "", "en_a": "", "en_b": ""},
        {"corpus": "daccord", "label": "contradiction", "category": "", "en_a": "", "en_b": ""},
        {"corpus": "daccord", "label": "contradiction", "category": "", "en_a": "", "en_b": ""},
        {"corpus": "breaking_fr", "label": "contradiction", "category": "antonyms", "en_a": "x", "en_b": "y"},
    ]


def test_report_scatters_scores_back_onto_their_own_pairs() -> None:
    # ``orig`` covers only the last pair: its single score must land there, not on pair 0.
    scores = {
        "v3-auglex-seed42": {
            "mt": [0.1, 0.2, 0.8, 0.9, 0.7],
            "mt_index": [0, 1, 2, 3, 4],
            "orig": [0.95],
            "orig_index": [4],
        },
        "offshelf-mdeberta-v3-base": {"fr": [0.9, 0.8, 0.2, 0.1, 0.3], "fr_index": [0, 1, 2, 3, 4]},
    }
    out = report(_pairs(), scores)
    daccord = out["corpora"]["daccord"]
    assert daccord["v3-auglex-ensemble/mt"]["auc"] == 1.0
    # A head that orders DACCORD backwards has AUC 0, and that is what the gap must show.
    assert daccord["offshelf-mdeberta-v3-base/fr"]["auc"] == 0.0
    assert "v3-auglex-ensemble/orig" not in daccord
    breaking = out["corpora"]["breaking_fr"]
    assert breaking["v3-auglex-ensemble/orig"]["per_family_below_zero"] == {"antonymes": 1.0}
    # 1a: the only multilingual head reading French is the bar, and a v3 head is never one.
    assert out["criterion_1"]["1a"]["reference"] == "offshelf-mdeberta-v3-base/fr"
    assert out["criterion_1"]["1a"]["gaps"]["v3-auglex-ensemble/mt"]["gap"] == 1.0
    # 1b: the best translate-test is the bar, and it is not compared with itself.
    assert out["criterion_1"]["1b"]["reference"] == "v3-auglex-ensemble/mt"
    assert out["criterion_1"]["1b"]["gaps"]["offshelf-mdeberta-v3-base/fr"]["gap"] == -1.0
    assert "v3-auglex-ensemble/mt" not in out["criterion_1"]["1b"]["gaps"]


def test_bar_1b_is_the_best_translate_test_not_the_first() -> None:
    scores = {
        "v3-auglex-seed42": {"mt": [0.6, 0.6, 0.4, 0.4, 0.5], "mt_index": [0, 1, 2, 3, 4]},
        "offshelf-deberta-v3-large": {"mt": [0.1, 0.2, 0.8, 0.9, 0.5], "mt_index": [0, 1, 2, 3, 4]},
    }
    out = report(_pairs(), scores)
    assert out["criterion_1"]["1b"]["reference"] == "offshelf-deberta-v3-large/mt"
    assert "1a" not in out["criterion_1"]


def test_breaking_families_score_synonyms_on_entailment_only() -> None:
    pairs = [
        {"corpus": "breaking_fr", "label": "entailment", "category": "synonyms", "en_a": "", "en_b": ""},
        {"corpus": "breaking_fr", "label": "contradiction", "category": "colors", "en_a": "", "en_b": ""},
        {"corpus": "breaking_fr", "label": "entailment", "category": "colors", "en_a": "", "en_b": ""},
    ]
    scores = {"offshelf-xlm-roberta-large": {"fr": [0.9, 0.9, 0.9], "fr_index": [0, 1, 2]}}
    family = report(pairs, scores)["corpora"]["breaking_fr"]["offshelf-xlm-roberta-large/fr"]["per_family_below_zero"]
    # The entailment-labelled colour pair is noise in the corpus, not a co-hyponym contradiction.
    assert family == {"co-hyponymes": 1.0, "synonymes": 1.0}


def test_tables_use_readable_names_two_decimals_and_mark_the_bars() -> None:
    scores = {
        "v3-auglex-seed42": {"mt": [0.1, 0.2, 0.8, 0.9, 0.7], "mt_index": [0, 1, 2, 3, 4]},
        "offshelf-xlm-roberta-large": {"fr": [0.3, 0.5, 0.6, 0.35, 0.6], "fr_index": [0, 1, 2, 3, 4]},
    }
    text = tables(report(_pairs(), scores))
    assert "| `v3 AUG+LEX` | traduit | 100.00 |" in text
    assert "| `XLM-R-large NLI` | français | 75.00 |" in text
    assert "1a |" in text and "1b |" in text
    assert "offshelf-" not in text and "/mt" not in text


def test_report_averages_seeds_into_one_ensemble() -> None:
    scores = {
        "v3-auglex-seed42": {"mt": [0.2, 0.2, 0.6, 0.6, 0.5], "mt_index": [0, 1, 2, 3, 4]},
        "v3-auglex-seed43": {"mt": [0.4, 0.4, 0.8, 0.8, 0.5], "mt_index": [0, 1, 2, 3, 4]},
    }
    out = report(_pairs(), scores)
    below = out["corpora"]["daccord"]["v3-auglex-ensemble/mt"]["below_zero_contradiction"]
    assert below == 1.0  # mean 0.7 on both contradictions
    assert out["v3_translate_test_daccord_auc_per_seed"]["std"] == 0.0


def test_each_trained_family_gets_its_own_ensemble_and_is_never_a_bar() -> None:
    scores = {
        "v4-xlmr-fr-seed42": {"fr": [0.1, 0.1, 0.9, 0.9, 0.9], "fr_index": [0, 1, 2, 3, 4]},
        "v4-xlmr-fr-seed43": {"fr": [0.3, 0.3, 0.7, 0.7, 0.7], "fr_index": [0, 1, 2, 3, 4]},
        "offshelf-xlm-roberta-large": {"fr": [0.4, 0.5, 0.6, 0.45, 0.6], "fr_index": [0, 1, 2, 3, 4]},
    }
    out = report(_pairs(), scores)
    assert out["seeds"] == {"v4-xlmr-fr": 2}
    assert "v4-xlmr-fr-ensemble/fr" in out["corpora"]["daccord"]
    # The trained French head is what is measured against the bar, never the bar itself.
    assert out["criterion_1"]["1a"]["reference"] == "offshelf-xlm-roberta-large/fr"
    assert "v4-xlmr-fr-ensemble/fr" in out["criterion_1"]["1a"]["gaps"]
    assert "| `XLM-R-large AUG+LEX (FR traduit)` | français |" in tables(out)


def test_xnli_rows_pair_french_with_its_english_item() -> None:
    rows = [
        {
            "premise": "Il pleut.",
            "hypothesis": "Il fait sec.",
            "premise_en": "It rains.",
            "hypothesis_en": "It is dry.",
            "label": "2",
        }
    ]
    pair = normalize("xnli_fr", rows)[0]
    assert (pair["label"], pair["en_a"], pair["fr_b"]) == ("contradiction", "It rains.", "Il fait sec.")


def test_delong_matches_auc_and_is_null_for_identical_heads() -> None:
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(1.0, 1.0, 300), rng.normal(0.0, 1.0, 300)
    same = delong(pos, neg, pos, neg)
    assert same["diff"] == 0.0 and same["p"] == 1.0
    # Ties count one half, as the plain AUC does.
    tied = delong(np.array([1.0, 0.5]), np.array([0.5, 0.0]), np.array([1.0, 0.5]), np.array([0.5, 0.0]))
    assert tied["auc_a"] == 0.875


def test_delong_detects_a_better_head_and_not_noise() -> None:
    rng = np.random.default_rng(2)
    pos, neg = rng.normal(1.0, 1.0, 400), rng.normal(0.0, 1.0, 400)
    better = delong(pos + 1.0, neg, pos, neg)
    assert better["diff"] > 0 and better["z"] > 3 and better["p"] < 0.001
    # Paired noise of a size the AUC barely moves on is not significant.
    jitter = delong(pos + rng.normal(0, 0.01, 400), neg, pos, neg)
    assert jitter["p"] > 0.05


def test_delong_standard_error_agrees_with_the_bootstrap() -> None:
    rng = np.random.default_rng(3)
    pos, neg = rng.normal(1.0, 1.0, 500), rng.normal(0.0, 1.0, 500)
    other = pos + rng.normal(0, 0.8, 500)
    test = delong(other, neg, pos, neg)
    boot = bootstrap_auc_gap(other, neg, pos, neg, rounds=1000)
    assert abs(test["se"] - (boot["high"] - boot["low"]) / (2 * 1.96)) < 0.3 * test["se"]


def test_mcnemar_counts_only_discordant_items() -> None:
    a = np.array([True, True, True, False, True, True])
    b = np.array([True, False, False, False, True, False])
    out = mcnemar(a, b)
    assert (out["only_a"], out["only_b"]) == (3, 0)
    assert out["p"] == 0.25  # exact two-sided binomial, 3 of 3
    assert mcnemar(a, a)["p"] == 1.0


def test_holm_is_monotone_and_capped() -> None:
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adjusted["a"] == 0.04
    assert adjusted["c"] == pytest.approx(0.09)
    assert adjusted["b"] == pytest.approx(0.09)  # 2 x 0.04 = 0.08, raised to keep the order
    assert adjusted["d"] == 0.5


def test_welch_needs_two_seeds_per_side() -> None:
    assert np.isnan(welch([0.8], [0.9, 0.91])["p"])
    assert welch([0.80, 0.81, 0.79], [0.90, 0.91, 0.89])["p"] < 0.01


def _pairs_with_guard_rails() -> list[dict]:
    daccord = [
        {"corpus": "daccord", "label": lab, "category": "", "en_a": "", "en_b": ""}
        for lab in ["compatible"] * 3 + ["contradiction"] * 3
    ]
    breaking = [
        {"corpus": "breaking_fr", "label": "contradiction", "category": "colors", "en_a": "x", "en_b": "y"}
        for _ in range(4)
    ]
    xnli = [
        {"corpus": "xnli_fr", "label": lab, "category": "", "en_a": "x", "en_b": "y"}
        for lab in ["entailment"] * 3 + ["contradiction"] * 3
    ]
    return daccord + breaking + xnli


def test_significance_compares_with_1a_1b_v3_and_writes_its_tables() -> None:
    pairs = _pairs_with_guard_rails()
    every = list(range(len(pairs)))
    good = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9] + [0.2] * 4 + [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    noisy = [0.4, 0.6, 0.3, 0.5, 0.7, 0.45] + [0.9] * 4 + [0.4, 0.6, 0.3, 0.5, 0.7, 0.45]
    scores = {
        "v3-auglex-seed42": {"mt": good, "mt_index": every, "fr": good, "fr_index": every},
        "v3-auglex-seed43": {"mt": good, "mt_index": every, "fr": good, "fr_index": every},
        "offshelf-xlm-roberta-large": {"fr": noisy, "fr_index": every, "orig": good[6:], "orig_index": every[6:]},
        "v4-xlmr-en-seed42": {"fr": noisy, "fr_index": every},
        "v4-xlmr-en-seed43": {"fr": good, "fr_index": every},
    }
    out = report(pairs, scores)
    sig = out["significance"]
    assert sig["references"] == {
        "1a": "offshelf-xlm-roberta-large/fr",
        "1b": "v3-auglex-ensemble/mt",
        "v3": "v3-auglex-ensemble/mt",
    }
    versus = sig["daccord"]["offshelf-xlm-roberta-large/fr"]
    assert "1a" not in versus and versus["v3"]["diff"] < 0 and "p_holm" in versus["v3"]
    assert versus["mcnemar_sign_v3"]["only_b"] >= 1
    # Guard rail: every colour contradiction of the off-the-shelf head falls below zero,
    # none is kept above it, so the binomial cannot clear 90 %.
    rail = sig["guard_rails"]["offshelf-xlm-roberta-large/fr"]["co-hyponymes"]
    assert rail["above_zero"] == 0.0 and rail["p_above_target"] == 1.0
    assert rail["mcnemar_v3"]["only_b"] == 4
    assert sig["seeds"]["v4-xlmr-en"]["n"] == 2 and "p" in sig["seeds"]["v4-xlmr-en"]["vs_v3"]
    parity = sig["parity"]["offshelf-xlm-roberta-large"]["xnli_fr"]
    assert parity["auc_en"] == 1.0 and parity["gap"] > 0
    text = tables(out)
    assert "## Significativité sur DACCORD" in text and "## Parité anglais / français" in text
    assert "| `XLM-R-large NLI` | XNLI-fr | 100.00 |" in text
