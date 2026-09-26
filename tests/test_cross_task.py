"""Tests for the cross-task evaluator (``src/diagnostics/cross_task.py``).

The point of this module is to put checkpoints with different output shapes on one table,
and the way that goes wrong is silently: a derived number presented as a measured one, or
a metric faked for a model that cannot produce it. So the tests are about what each shape
is allowed to answer, and about the arithmetic behind the two shared axes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from data.schema import POLARITY_CLASSES
from diagnostics.cross_task import CLASS_NAMES, auc, macro_f1

ENTAIL = POLARITY_CLASSES["entailment"]
NEUTRAL = POLARITY_CLASSES["neutral"]
CONTRA = POLARITY_CLASSES["contradiction"]


# --- the shared axis: AUC ------------------------------------------------------------


def test_perfect_ordering_scores_one():
    assert auc(np.array([9.0, 8.0]), np.array([2.0, 1.0])) == pytest.approx(1.0)


def test_perfectly_inverted_ordering_scores_zero():
    # The signature of a swapped label mapping, which this project has already shipped once.
    assert auc(np.array([1.0, 2.0]), np.array([8.0, 9.0])) == pytest.approx(0.0)


def test_two_classes_that_never_differ_score_one_half():
    # A clamped head saturates at exactly 0 and 100, so ties between the classes are the
    # common case. Ordinal ranks would answer 0.0 here, which reads as a perfectly inverted
    # model when the truth is that it cannot discriminate.
    tied = np.full(20, 100.0)
    assert auc(tied, tied) == pytest.approx(0.5)


def test_a_tie_counts_as_half_a_correctly_ordered_couple():
    assert auc(np.array([90.0, 50.0]), np.array([50.0])) == pytest.approx(0.75)


def test_an_empty_class_reports_nan_rather_than_a_confident_number():
    assert math.isnan(auc(np.array([1.0]), np.array([])))


# --- the shared axis: macro-F1 --------------------------------------------------------


def test_a_perfect_classifier_scores_one():
    labels = np.array([ENTAIL, NEUTRAL, CONTRA])
    assert macro_f1(labels.copy(), labels) == pytest.approx(1.0)


def test_a_class_never_predicted_drags_the_macro_down():
    # The failure macro-F1 exists to catch: a head that refuses a minority class still
    # scores respectably on accuracy.
    labels = np.array([ENTAIL] * 7 + [NEUTRAL] * 3)
    predictions = np.array([ENTAIL] * 10)
    assert macro_f1(predictions, labels) < 0.7 * (10 / 10)
    assert macro_f1(predictions, labels) == pytest.approx(2 * 0.7 / 1.7 / 3)


def test_macro_f1_is_the_unweighted_mean_over_the_three_classes():
    labels = np.array([ENTAIL, ENTAIL, NEUTRAL, CONTRA])
    predictions = np.array([ENTAIL, NEUTRAL, NEUTRAL, CONTRA])
    # entailment F1 = 2*1*0.5/1.5 = 2/3 ; neutral = 2*0.5*1/1.5 = 2/3 ; contradiction = 1
    assert macro_f1(predictions, labels) == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_the_class_order_follows_the_schema():
    # Every index here is read positionally, so a name list out of step with the schema
    # would relabel the whole table without changing a number.
    assert CLASS_NAMES == ("entailment", "neutral", "contradiction")
    assert [POLARITY_CLASSES[name] for name in CLASS_NAMES] == [0, 1, 2]
