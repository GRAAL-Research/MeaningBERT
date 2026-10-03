"""Tests for the per-corpus and ranking breakdown (``src/diagnostics/head_breakdown.py``).

The module exists to answer two questions the aggregate hides: whether the headline comes
from one corpus, and whether the head orders ``p_contra`` well. Both answers are a single
number that looks plausible whatever the code does, so the tests pin the two places a
wrong number can come from silently: the quantity the AUC is computed on, and the slicing
that produces a per-corpus figure.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from datasets import Dataset

from data.schema import POLARITY_CLASSES
from diagnostics.head_breakdown import breakdown, ranking_score, slice_metrics, softmax

ENTAILMENT = POLARITY_CLASSES["entailment"]
NEUTRAL = POLARITY_CLASSES["neutral"]
CONTRADICTION = POLARITY_CLASSES["contradiction"]


def logits_for(classes, confidence: float = 4.0) -> np.ndarray:
    """Logits whose argmax is the given class, with a tunable margin."""
    out = np.zeros((len(classes), 3))
    for row, index in enumerate(classes):
        out[row, index] = confidence
    return out


class TestRankingScore:
    def test_it_is_one_minus_the_contradiction_probability(self):
        """The composition multiplies the magnitude by exactly this quantity.

        An AUC computed on the entailment probability instead would measure a number the
        signed scale never sees, and would differ whenever the neutral mass moves.
        """
        logits = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]])

        got = ranking_score(logits)

        assert got[0] == pytest.approx(1 - softmax(logits)[0, CONTRADICTION])
        assert got[0] < 0.01
        assert got[1] > 0.99

    def test_it_ignores_how_the_rest_of_the_mass_is_split(self):
        """Two heads equally sure of 'not a contradiction' must score the same."""
        entail_heavy = np.array([[6.0, 0.0, -6.0]])
        neutral_heavy = np.array([[0.0, 6.0, -6.0]])

        assert ranking_score(entail_heavy)[0] == pytest.approx(ranking_score(neutral_heavy)[0])


class TestSliceMetrics:
    def test_a_perfect_head_scores_one_everywhere(self):
        truth = np.array([ENTAILMENT, NEUTRAL, CONTRADICTION] * 4)
        got = slice_metrics(logits_for(truth), truth)

        assert got["macro_f1"] == pytest.approx(1.0)
        assert got["accuracy"] == pytest.approx(1.0)
        assert got["auc"] == pytest.approx(1.0)

    def test_a_head_that_prefers_contradictions_scores_below_chance(self):
        """An AUC under one half is a metric that ranks the wrong class first."""
        truth = np.array([ENTAILMENT, ENTAILMENT, CONTRADICTION, CONTRADICTION])
        reversed_logits = logits_for([CONTRADICTION, CONTRADICTION, ENTAILMENT, ENTAILMENT])

        assert slice_metrics(reversed_logits, truth)["auc"] == pytest.approx(0.0)

    def test_per_class_recall_is_reported_for_every_class(self):
        truth = np.array([ENTAILMENT, ENTAILMENT, NEUTRAL, CONTRADICTION])
        predicted = logits_for([ENTAILMENT, NEUTRAL, NEUTRAL, CONTRADICTION])

        got = slice_metrics(predicted, truth)["per_class"]

        assert got["entailment"]["recall"] == pytest.approx(0.5)
        assert got["neutral"]["recall"] == pytest.approx(1.0)
        assert got["contradiction"]["n"] == 1

    def test_an_empty_slice_gives_nan_rather_than_zero(self):
        """A corpus absent from a split must not be reported as a model scoring nothing."""
        got = slice_metrics(np.zeros((0, 3)), np.array([], dtype=int))

        assert got["n"] == 0
        assert math.isnan(got["macro_f1"])


class _StubModel:
    """A head whose answer is decided by the sentence, so the slicing is observable."""

    def logits(self, left, right):  # noqa: ARG002 - the right side is unused on purpose
        return logits_for([ENTAILMENT if text.startswith("good") else CONTRADICTION for text in left])


class TestBreakdown:
    def test_each_corpus_is_scored_on_its_own_rows(self):
        """One corpus perfect and one corpus wrong must not average into one number."""
        rows = Dataset.from_dict(
            {
                "original": ["good a", "good b", "bad c", "bad d"],
                "simplification": ["x"] * 4,
                "polarity": [float(ENTAILMENT), float(ENTAILMENT), float(ENTAILMENT), float(ENTAILMENT)],
                "corpus": ["sick", "sick", "vitaminc", "vitaminc"],
            }
        )

        got = breakdown(_StubModel(), rows)

        assert got["per_corpus"]["sick"]["accuracy"] == pytest.approx(1.0)
        assert got["per_corpus"]["vitaminc"]["accuracy"] == pytest.approx(0.0)
        assert got["overall"]["accuracy"] == pytest.approx(0.5)

    def test_the_corpus_shares_add_up(self):
        rows = Dataset.from_dict(
            {
                "original": ["good a", "good b", "bad c"],
                "simplification": ["x"] * 3,
                "polarity": [float(ENTAILMENT)] * 3,
                "corpus": ["sick", "sick", "vitaminc"],
            }
        )

        shares = breakdown(_StubModel(), rows)["corpus_shares"]

        assert sum(shares.values()) == pytest.approx(1.0)
        assert shares["sick"] == pytest.approx(2 / 3, abs=1e-4)
