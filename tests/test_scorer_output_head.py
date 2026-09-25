"""Tests for the output-head mapping in the public inference API.

``to_percent`` is the function that decides whether a published score is on the 0-100
scale or is quietly off it, and ``head_of`` is what picks which rule applies. Both sit on
the path every external user of the metric takes, and a mistake in either returns numbers
that look entirely reasonable. They were the only uncovered lines of the module, and the
symmetry tests monkeypatch ``to_percent`` away, so nothing exercised them.

The failure this guards against is concrete and has already been shipped once: the v1
model card prints ``logits.tolist()`` directly, which on a v2 clamped checkpoint returns
unit-scale values, so a 0.83 pair reads as a score of 0.83 instead of 83.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the output head is a tensor operation")

from meaningbert.scorer import DEFAULT_HEAD, SCORE_MAX, head_of, to_percent  # noqa: E402


class _Config:
    """Stand-in for a loaded PretrainedConfig."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _scores(values, head):
    return to_percent(torch.tensor(values, dtype=torch.float32), head).tolist()


# --- head_of -------------------------------------------------------------------------


def test_a_v1_checkpoint_without_the_field_is_read_as_linear():
    # v1 predates the field entirely. Defaulting to anything else would rescale scores
    # that are already on the 0-100 scale.
    assert head_of(_Config()) == "linear"
    assert DEFAULT_HEAD == "linear"


def test_a_v2_checkpoint_declares_its_own_head():
    assert head_of(_Config(meaningbert_output_head="clamped")) == "clamped"


def test_an_empty_head_field_falls_back_instead_of_returning_nothing():
    # An empty string would reach to_percent and raise on an unknown head, which is a
    # confusing way to say "this is a v1 checkpoint".
    assert head_of(_Config(meaningbert_output_head="")) == "linear"
    assert head_of(_Config(meaningbert_output_head=None)) == "linear"


# --- to_percent: one rule per head ---------------------------------------------------


def test_a_linear_head_treats_the_logit_as_the_score_already():
    assert _scores([0.0, 42.0, 100.0], "linear") == pytest.approx([0.0, 42.0, 100.0])


def test_a_clamped_head_multiplies_the_unit_scale_by_one_hundred():
    # This is the published v2 checkpoint. Reading the logit directly, as the v1 card
    # shows, would report 0.83 where the score is 83.
    assert _scores([0.0, 0.83, 1.0], "clamped") == pytest.approx([0.0, 83.0, 100.0])


def test_normalized_is_the_same_rule_as_clamped():
    assert _scores([0.42], "normalized") == pytest.approx(_scores([0.42], "clamped"))


def test_a_sigmoid_head_squashes_before_scaling():
    assert _scores([0.0], "sigmoid") == pytest.approx([50.0])
    # The ceiling that motivated replacing this head in v2: an open interval cannot reach
    # 100, so an identical pair is capped below the sanity-check threshold.
    assert _scores([10.0], "sigmoid")[0] < SCORE_MAX


def test_an_unknown_head_raises_instead_of_guessing():
    with pytest.raises(ValueError, match="plausible wrong scores"):
        _scores([0.5], "softmax")


# --- clipping and shape --------------------------------------------------------------


def test_a_linear_head_below_zero_is_clipped_not_returned_negative():
    # v1's head is unbounded and does emit negatives. A negative meaning-preservation
    # score has no reading on the v2 scale.
    assert _scores([-12.0], "linear") == pytest.approx([0.0])


def test_a_clamped_head_above_one_is_clipped_to_one_hundred():
    assert _scores([1.4], "clamped") == pytest.approx([100.0])


def test_a_column_shaped_logit_is_flattened_to_one_score_per_pair():
    # The model returns (n, 1). Without the squeeze the caller gets a nested list and the
    # pairing with the input silently shifts.
    got = to_percent(torch.tensor([[0.5], [0.25]], dtype=torch.float32), "clamped").tolist()
    assert got == pytest.approx([50.0, 25.0])


def test_the_two_shapes_give_the_same_scores():
    flat = to_percent(torch.tensor([0.5, 0.25]), "clamped").tolist()
    column = to_percent(torch.tensor([[0.5], [0.25]]), "clamped").tolist()
    assert flat == pytest.approx(column)
