"""Tests of the bounded heads (C3) on the torch side.

The scale helpers are backend agnostic: the same functions run on numpy arrays in
``tests/test_calibration.py`` and on torch tensors here. These tests are skipped on a
machine without torch, and run in CI and on the training server, which both have it.
"""

import pytest

from training.calibration import (
    SCORE_MAX,
    SCORE_MIN,
    clip_to_score_range,
    percent_from_logits,
    targets_for_head,
    unit_from_logits,
)

torch = pytest.importorskip("torch", reason="torch is not installed on this machine.")

EXTREME_LOGITS = [-1e30, -1e6, -800.0, -50.0, -1.0, 0.0, 1.0, 50.0, 800.0, 1e6, 1e30]


class TestBoundedHeadsOnTensors:
    def test_sigmoid_head_stays_in_range_for_extreme_logits(self):
        scores = percent_from_logits(torch.tensor(EXTREME_LOGITS, dtype=torch.float32), head="sigmoid")

        assert torch.all(torch.isfinite(scores))
        assert float(scores.min()) >= SCORE_MIN
        assert float(scores.max()) <= SCORE_MAX

    def test_sigmoid_head_saturates_on_both_sides(self):
        scores = percent_from_logits(torch.tensor([-1e30, 0.0, 1e30]), head="sigmoid")

        assert float(scores[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(scores[1]) == pytest.approx(50.0)
        assert float(scores[2]) == pytest.approx(100.0, abs=1e-6)

    def test_normalized_head_stays_in_range_for_extreme_logits(self):
        scores = percent_from_logits(torch.tensor([-1e6, -0.5, 0.0, 0.42, 1.0, 1e6]), head="normalized")

        assert float(scores.min()) >= SCORE_MIN
        assert float(scores.max()) <= SCORE_MAX
        assert float(scores[3]) == pytest.approx(42.0, abs=1e-4)

    def test_linear_head_is_left_untouched(self):
        logits = torch.tensor([-40.0, 0.0, 250.0])

        assert torch.equal(percent_from_logits(logits, head="linear"), logits)

    def test_clip_to_score_range_on_a_tensor(self):
        clipped = clip_to_score_range(torch.tensor([-10.0, 42.0, 140.0]))

        assert clipped.tolist() == [0.0, 42.0, 100.0]

    def test_gradients_flow_through_the_sigmoid_head(self):
        logits = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
        labels = torch.tensor([10.0, 50.0, 90.0])

        predictions = unit_from_logits(logits, "sigmoid")
        loss = torch.nn.functional.mse_loss(predictions, targets_for_head(labels, "sigmoid"))
        loss.backward()

        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad))
        assert float(logits.grad.abs().sum()) > 0


class TestPreprocessHook:
    """The hook that puts evaluation predictions back on the 0-100 scale."""

    @staticmethod
    def _hook(head):
        few_shot_training = pytest.importorskip(
            "training.few_shot_training", reason="transformers or wandb is not installed on this machine."
        )
        return few_shot_training.make_logits_to_percent(head)

    def test_linear_head_needs_no_hook(self):
        assert self._hook("linear") is None

    def test_sigmoid_hook_returns_scores_in_range(self):
        hook = self._hook("sigmoid")
        scores = hook(torch.tensor(EXTREME_LOGITS, dtype=torch.float32), torch.zeros(len(EXTREME_LOGITS)))

        assert float(scores.min()) >= SCORE_MIN
        assert float(scores.max()) <= SCORE_MAX

    def test_normalized_hook_returns_scores_in_range(self):
        hook = self._hook("normalized")
        scores = hook(torch.tensor([-5.0, 0.3, 7.0]), torch.zeros(3))

        assert float(scores.min()) >= SCORE_MIN
        assert float(scores.max()) <= SCORE_MAX
        assert float(scores[1]) == pytest.approx(30.0, abs=1e-4)

    def test_hook_accepts_a_tuple_of_logits(self):
        hook = self._hook("sigmoid")
        scores = hook((torch.tensor([0.0, 0.0]), torch.tensor([1.0])), torch.zeros(2))

        assert scores.tolist() == [50.0, 50.0]
