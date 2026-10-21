"""Tests for the French magnitude check (``src/diagnostics/magnitude_fr.py``).

V4-H3 is decided by one Pearson against BERTScore. The two places that number can drift
silently are the BERTScore formula itself and the bootstrap interval around the Pearson.
"""

from __future__ import annotations

import numpy as np

from diagnostics.magnitude_fr import BAR, compare, correlation, greedy_f1, markdown, trained_models
from diagnostics.significance import williams


def test_greedy_f1_is_one_for_identical_embeddings_and_order_free() -> None:
    tokens = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert greedy_f1(tokens, tokens) == 1.0
    assert greedy_f1(tokens[::-1], tokens) == 1.0


def test_greedy_f1_drops_for_orthogonal_tokens() -> None:
    assert greedy_f1(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])) == 0.0


def test_greedy_f1_balances_precision_and_recall() -> None:
    # The candidate covers one of the two reference tokens: precision 1, recall 0.5.
    candidate = np.array([[1.0, 0.0]])
    reference = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert abs(greedy_f1(candidate, reference) - 2 * 1 * 0.5 / 1.5) < 1e-9


def test_correlation_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    gold = rng.normal(size=200)
    predicted = gold + rng.normal(scale=0.5, size=200)
    out = correlation(predicted, gold, rounds=300)
    assert out["pearson_low"] <= out["pearson"] <= out["pearson_high"]
    assert out["pearson"] > 0.8 and out["n"] == 200


def test_williams_is_null_for_equal_correlations_and_antisymmetric() -> None:
    assert williams(0.5, 0.5, 0.7, 100)["t"] == 0.0
    assert williams(0.5, 0.5, 0.7, 100)["p"] == 1.0
    forward, backward = williams(0.6, 0.4, 0.5, 200), williams(0.4, 0.6, 0.5, 200)
    assert forward["t"] == -backward["t"] and forward["p"] == backward["p"]


def test_williams_gains_power_when_the_metrics_agree() -> None:
    # The same gap in correlation is more significant between two metrics that agree with
    # each other, which is the point of a test for dependent correlations.
    assert williams(0.6, 0.5, 0.9, 150)["p"] < williams(0.6, 0.5, 0.1, 150)["p"]


def test_compare_tests_each_candidate_against_bertscore_only() -> None:
    rng = np.random.default_rng(4)
    gold = rng.normal(size=300)
    scores = {
        BAR: gold + rng.normal(scale=1.0, size=300),
        "fort": gold + rng.normal(scale=0.3, size=300),
        "bruit": rng.normal(size=300),
    }
    out = compare(scores, gold)
    assert set(out) == {"fort", "bruit"}
    assert out["fort"]["diff"] > 0 and out["fort"]["p_holm"] < 0.01
    assert out["bruit"]["diff"] < 0 and out["bruit"]["p_holm"] < 0.01


def test_trained_models_group_seeds_by_backbone_and_corpus(tmp_path) -> None:
    for family, seed in (("mdeberta-fr", 42), ("mdeberta-fr", 43), ("xlmr-en", 42)):
        folder = tmp_path / family / f"microsoft_mdeberta-v3-base_d_full-fr_clamped_seed{seed}"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}")
    (tmp_path / "mdeberta-fr" / "checkpoint-100").mkdir()
    families = trained_models(str(tmp_path))
    assert sorted(families) == ["mdeberta-fr", "xlmr-en"]
    assert len(families["mdeberta-fr"]) == 2


def test_markdown_ranks_by_pearson_and_marks_the_bar() -> None:
    block = {"pearson": 0.5, "pearson_low": 0.4, "pearson_high": 0.6, "spearman": 0.5, "n": 10}
    result = {
        "candidates": {BAR: {"all": block, "test": block}, "v4": {"all": {**block, "pearson": 0.7}, "test": block}},
        "versus_bar": {"v4": {"diff": 0.2, "t": 3.0, "p": 0.001, "p_holm": 0.001}},
    }
    text = markdown(result)
    assert text.index("| v4 |") < text.index(f"| {BAR} |")
    assert "+0.20" in text and "< 0.01" in text and "réf." in text
