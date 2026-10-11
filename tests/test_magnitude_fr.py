"""Tests for the French magnitude check (``src/diagnostics/magnitude_fr.py``).

V4-H3 is decided by one Pearson against BERTScore. The two places that number can drift
silently are the BERTScore formula itself and the bootstrap interval around the Pearson.
"""

from __future__ import annotations

import numpy as np

from diagnostics.magnitude_fr import correlation, greedy_f1


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
