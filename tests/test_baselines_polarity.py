"""Tests for the baselines that bound the polarity head (``src/diagnostics/baselines_polarity.py``).

A baseline that is quietly broken flatters the model it is there to bound, and nothing in
the pipeline would say so: it would simply print a low number, which is what a baseline is
expected to print. The tests therefore check that each baseline reaches what its own
construction guarantees on data built to be separable, and refuses to on data built not
to be.
"""

from __future__ import annotations

import math

import pytest
from datasets import Dataset

from diagnostics.baselines_polarity import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    auc_entail_vs_contra,
    majority_baseline,
    overlap,
    overlap_baseline,
    tfidf_baseline,
)
import numpy as np


def rows(items) -> Dataset:
    return Dataset.from_dict({
        "original": [o for o, _, _ in items],
        "simplification": [s for _, s, _ in items],
        "polarity": [float(p) for _, _, p in items],
    })


class TestOverlap:
    def test_identical_sentences_overlap_completely(self):
        assert overlap("a dog is running", "a dog is running") == pytest.approx(1.0)

    def test_disjoint_sentences_do_not_overlap(self):
        assert overlap("a dog is running", "quantum field theory") == pytest.approx(0.0)

    def test_the_measure_is_symmetric(self):
        """Jaccard and not containment: a magnitude metric has no direction either."""
        left, right = "a dog is running fast", "a dog is running"

        assert overlap(left, right) == pytest.approx(overlap(right, left))

    def test_two_empty_sentences_give_zero_rather_than_dividing_by_zero(self):
        assert overlap("", "") == 0.0

    def test_case_and_punctuation_do_not_change_the_measure(self):
        assert overlap("A Dog, running!", "a dog running") == pytest.approx(1.0)


class TestMajority:
    def test_it_predicts_the_most_frequent_training_class(self):
        train = rows([("a", "b", CONTRADICTION)] * 5 + [("c", "d", ENTAILMENT)] * 2)
        test = rows([("e", "f", ENTAILMENT), ("g", "h", CONTRADICTION)])

        got = majority_baseline(train, test)

        assert got["predicted_class"] == CONTRADICTION
        assert got["accuracy"] == pytest.approx(0.5)

    def test_on_a_balanced_split_it_lands_on_the_arithmetic_floor(self):
        """Three balanced classes, one guess: a third of accuracy and a sixth of macro-F1."""
        train = rows([("a", "b", ENTAILMENT)] * 3)
        test = rows([("a", "b", c) for c in (ENTAILMENT, NEUTRAL, CONTRADICTION)] * 4)

        got = majority_baseline(train, test)

        assert got["accuracy"] == pytest.approx(1 / 3)
        assert got["macro_f1"] == pytest.approx(1 / 6, abs=1e-6)


class TestAuc:
    def test_a_score_that_ranks_entailment_first_reaches_one(self):
        truth = np.array([ENTAILMENT, ENTAILMENT, CONTRADICTION, CONTRADICTION])
        assert auc_entail_vs_contra(np.array([0.9, 0.8, 0.2, 0.1]), truth) == pytest.approx(1.0)

    def test_a_reversed_score_reaches_zero(self):
        """Below one half is not noise: it is a metric that prefers the contradiction."""
        truth = np.array([ENTAILMENT, ENTAILMENT, CONTRADICTION, CONTRADICTION])
        assert auc_entail_vs_contra(np.array([0.1, 0.2, 0.8, 0.9]), truth) == pytest.approx(0.0)

    def test_neutral_pairs_are_excluded_rather_than_counted_as_contradictions(self):
        """Neutral rows carry no polarity to rank, and counting them would move the number."""
        truth = np.array([ENTAILMENT, CONTRADICTION, NEUTRAL, NEUTRAL])
        score = np.array([0.9, 0.1, 0.0, 1.0])

        assert auc_entail_vs_contra(score, truth) == pytest.approx(1.0)

    def test_a_split_with_only_one_of_the_two_classes_gives_nan(self):
        truth = np.array([ENTAILMENT, ENTAILMENT, NEUTRAL])
        assert math.isnan(auc_entail_vs_contra(np.array([0.1, 0.9, 0.5]), truth))


class TestOverlapBaseline:
    def test_it_separates_classes_that_overlap_tells_apart(self):
        """Built so the ordering the baseline assumes is the true one.

        Neutral pairs share nothing, contradictions share a little, entailments share a
        lot. If the thresholds are fitted correctly the baseline should be well above the
        0.167 macro-F1 floor on data shaped like its own assumption.
        """
        def sample(index):
            return [
                (f"alpha beta gamma delta {index}", f"zeta eta theta iota {index}", NEUTRAL),
                (f"alpha beta gamma delta {index}", f"alpha beta kappa lambda {index}", CONTRADICTION),
                (f"alpha beta gamma delta {index}", f"alpha beta gamma delta {index}", ENTAILMENT),
            ]
        data = rows([r for i in range(40) for r in sample(i)])

        got = overlap_baseline(data, data, data)

        assert got["macro_f1"] > 0.8
        assert got["low"] < got["high"]

    def test_it_stays_near_the_floor_when_overlap_carries_no_signal(self):
        """Every pair has the same overlap, so no threshold can separate anything."""
        data = rows([(f"alpha beta {i}", f"alpha beta {i}", c)
                     for i in range(40) for c in (ENTAILMENT, NEUTRAL, CONTRADICTION)])

        assert overlap_baseline(data, data, data)["macro_f1"] < 0.25


class TestTfidfBaseline:
    def test_it_learns_a_lexical_rule_that_is_actually_there(self):
        data = rows([(f"the subject {i} acts", f"the subject {i} {word}", cls)
                     for i in range(40)
                     for word, cls in (("acts", ENTAILMENT), ("elsewhere", NEUTRAL),
                                       ("never", CONTRADICTION))])

        got = tfidf_baseline(data, data)

        assert got["macro_f1"] > 0.9
        assert got["auc_entailment_vs_contradiction"] > 0.9

    def test_it_cannot_learn_a_label_the_tokens_do_not_carry(self):
        """Same two sentences under all three labels: a bag of words has nothing to go on."""
        data = rows([("alpha beta", "alpha beta", c)
                     for _ in range(40) for c in (ENTAILMENT, NEUTRAL, CONTRADICTION)])

        assert tfidf_baseline(data, data)["macro_f1"] < 0.25
