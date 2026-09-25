"""Tests for the polarity head's metrics (``src/training/train_polarity.py``).

Only the pure reporting half is tested here; the training loop is a thin wrapper over the
Trainer and testing it would mean testing transformers. What is worth testing is what the
experiment will be read from: the confusion matrix, whose orientation decides whether a
result says "the head calls contradictions neutral" or the opposite, and macro-F1, which
is the only number that notices a head refusing to predict a minority class.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from data.schema import POLARITY_CLASSES
from training.train_polarity import CLASS_NAMES, compute_metrics, confusion, per_class_f1, summarise

ENTAIL = POLARITY_CLASSES["entailment"]
NEUTRAL = POLARITY_CLASSES["neutral"]
CONTRA = POLARITY_CLASSES["contradiction"]


def _arrays(pairs):
    """From (true, predicted) pairs to the two arrays the reporters take."""
    return np.array([p for _, p in pairs]), np.array([t for t, _ in pairs])


# --- class order ---------------------------------------------------------------------


def test_the_class_names_follow_the_schema_indices():
    # Every matrix row and column is read by position, so a name list out of step with the
    # indices would relabel the whole report without changing a number.
    assert CLASS_NAMES == ("entailment", "neutral", "contradiction")
    assert [POLARITY_CLASSES[name] for name in CLASS_NAMES] == [0, 1, 2]


# --- the confusion matrix ------------------------------------------------------------


def test_a_perfect_classifier_fills_only_the_diagonal():
    predictions, labels = _arrays([(ENTAIL, ENTAIL), (NEUTRAL, NEUTRAL), (CONTRA, CONTRA)])
    assert confusion(predictions, labels) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_the_matrix_reads_row_true_column_predicted():
    # The orientation IS the finding. Calling a contradiction neutral costs the sign;
    # calling a neutral pair a contradiction invents one. A transposed matrix reports the
    # opposite failure with the same numbers.
    predictions, labels = _arrays([(CONTRA, NEUTRAL)])
    matrix = confusion(predictions, labels)
    assert matrix[CONTRA][NEUTRAL] == 1
    assert matrix[NEUTRAL][CONTRA] == 0


def test_every_prediction_lands_somewhere_in_the_matrix():
    pairs = [(ENTAIL, NEUTRAL), (NEUTRAL, NEUTRAL), (CONTRA, ENTAIL), (CONTRA, CONTRA)]
    predictions, labels = _arrays(pairs)
    assert sum(sum(row) for row in confusion(predictions, labels)) == len(pairs)


# --- F1 ------------------------------------------------------------------------------


def test_a_class_never_predicted_scores_zero_not_undefined():
    # The failure macro-F1 exists to catch: VitaminC's test split is 17 % neutral, so a
    # head that never predicts neutral still reaches a respectable accuracy.
    pairs = [(NEUTRAL, ENTAIL)] * 3 + [(ENTAIL, ENTAIL)] * 7
    predictions, labels = _arrays(pairs)
    got = summarise(predictions, labels)
    assert got["f1"]["neutral"] == 0.0
    assert got["accuracy"] == pytest.approx(0.7)
    assert got["macro_f1"] < got["accuracy"]


def test_macro_f1_is_the_unweighted_mean_over_the_three_classes():
    matrix = [[2, 0, 0], [0, 1, 1], [0, 0, 2]]
    scores = per_class_f1(matrix)
    assert scores["macro"] == pytest.approx(
        sum(scores[name] for name in CLASS_NAMES) / 3
    )
    assert scores["entailment"] == pytest.approx(1.0)


def test_f1_balances_precision_against_recall():
    # Predicted contradiction 4 times, right twice; 2 of 3 true contradictions found.
    # precision 0.5, recall 2/3, F1 = 2*0.5*(2/3)/(0.5+2/3) = 4/7.
    matrix = [[0, 0, 2], [0, 0, 0], [1, 0, 2]]
    assert per_class_f1(matrix)["contradiction"] == pytest.approx(4 / 7)


def test_a_perfect_classifier_scores_one_everywhere():
    got = summarise(*_arrays([(ENTAIL, ENTAIL), (NEUTRAL, NEUTRAL), (CONTRA, CONTRA)]))
    assert got["accuracy"] == pytest.approx(1.0)
    assert got["macro_f1"] == pytest.approx(1.0)


def test_an_empty_evaluation_reports_nan_rather_than_a_perfect_score():
    # Zero correct out of zero must not read as 100 %, and it must not crash a run that
    # has already paid for its training.
    got = summarise(np.array([]), np.array([]))
    assert math.isnan(got["accuracy"])
    assert got["n"] == 0


# --- the Trainer hook ----------------------------------------------------------------


def test_the_trainer_hook_turns_logits_into_the_two_scalars_it_tracks():
    # The Trainer hands logits, not classes. Forgetting the argmax would compare raw
    # scores to class indices and report a plausible, meaningless accuracy.
    logits = np.array([[9.0, 0.0, 0.0], [0.0, 0.0, 9.0], [0.0, 9.0, 0.0]])
    got = compute_metrics((logits, np.array([ENTAIL, CONTRA, NEUTRAL])))
    assert got == {"accuracy": pytest.approx(1.0), "macro_f1": pytest.approx(1.0)}


def test_the_trainer_hook_notices_a_wrong_prediction():
    logits = np.array([[9.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    got = compute_metrics((logits, np.array([ENTAIL, CONTRA])))
    assert got["accuracy"] == pytest.approx(0.5)


# --- reusing a pretrained NLI head, and refusing to reuse it wrong --------------------

from training.train_polarity import head_reuse_plan  # noqa: E402


def test_a_checkpoint_already_in_our_order_is_reused_untouched():
    # MoritzLaurer's NLI models publish exactly this order, so the head is reused as is and
    # the run starts from a model that already does the task.
    reusable, permutation = head_reuse_plan({0: "entailment", 1: "neutral", 2: "contradiction"})
    assert reusable
    assert permutation == [0, 1, 2]


def test_a_reversed_checkpoint_is_reused_but_permuted():
    # roberta-large-mnli publishes the reverse. Three labels either way, so nothing
    # mismatches and the head is silently kept: this is the same failure as the swapped
    # SICK encoding, wrong without ever crashing.
    reusable, permutation = head_reuse_plan({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"})
    assert reusable
    assert permutation == [2, 1, 0]


def test_the_permutation_maps_the_checkpoint_rows_into_our_order():
    _, permutation = head_reuse_plan({0: "contradiction", 1: "neutral", 2: "entailment"})
    reordered = [["row_contra"], ["row_neutral"], ["row_entail"]]
    got = [reordered[index] for index in permutation]
    assert got == [["row_entail"], ["row_neutral"], ["row_contra"]]


def test_a_plain_encoder_has_nothing_to_reuse():
    # bert-base-uncased and microsoft/deberta-v3-large publish LABEL_0 / LABEL_1: not an
    # inference label space, so the head is replaced and nothing is warned about.
    assert head_reuse_plan({0: "LABEL_0", 1: "LABEL_1"}) == (False, None)
    assert head_reuse_plan({0: "LABEL_0"}) == (False, None)
    assert head_reuse_plan(None) == (False, None)


def test_a_three_class_head_that_is_not_an_inference_space_is_left_alone():
    assert head_reuse_plan({0: "positive", 1: "negative", 2: "mixed"}) == (False, None)


def test_a_duplicated_class_name_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="not a permutation"):
        head_reuse_plan({0: "entailment", 1: "entailment", 2: "contradiction"})


def test_the_case_of_the_published_labels_does_not_matter():
    assert head_reuse_plan({0: "ENTAILMENT", 1: "Neutral", 2: " contradiction "})[1] == [0, 1, 2]
