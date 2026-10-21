"""Tests for the WiSE-FT interpolation of the v4 heads (``interpolate_heads.py``).

What would go wrong silently is an average of two heads whose rows mean different classes,
or of integer index buffers; both produce a model that loads and scores, wrongly.
"""

from __future__ import annotations

import pytest
import torch

from training.interpolate_heads import aligned_base, interpolate


def test_endpoints_and_midpoint() -> None:
    base = {"w": torch.zeros(2, 2), "ids": torch.arange(3)}
    tuned = {"w": torch.ones(2, 2) * 4, "ids": torch.arange(3) + 7}
    assert torch.equal(interpolate(base, tuned, 0.0)["w"], base["w"])
    assert torch.equal(interpolate(base, tuned, 1.0)["w"], tuned["w"])
    assert torch.allclose(interpolate(base, tuned, 0.25)["w"], torch.ones(2, 2))


def test_integer_buffers_come_from_the_tuned_head_untouched() -> None:
    base = {"ids": torch.arange(3)}
    tuned = {"ids": torch.tensor([5, 6, 7])}
    assert torch.equal(interpolate(base, tuned, 0.5)["ids"], tuned["ids"])


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_alpha_outside_unit_interval_is_refused(alpha: float) -> None:
    with pytest.raises(ValueError):
        interpolate({"w": torch.zeros(1)}, {"w": torch.zeros(1)}, alpha)


def test_mismatched_state_dicts_are_refused() -> None:
    with pytest.raises(ValueError):
        interpolate({"w": torch.zeros(1)}, {"v": torch.zeros(1)}, 0.5)
    with pytest.raises(ValueError):
        interpolate({"w": torch.zeros(2)}, {"w": torch.zeros(3)}, 0.5)


def test_base_head_rows_are_put_in_the_v4_order(tmp_path) -> None:
    from transformers import BertConfig, BertForSequenceClassification

    # Contradiction first, as joeddav/xlm-roberta-large-xnli publishes it.
    config = BertConfig(
        vocab_size=50,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=3,
        id2label={0: "contradiction", 1: "neutral", 2: "entailment"},
        label2id={"contradiction": 0, "neutral": 1, "entailment": 2},
    )
    model = BertForSequenceClassification(config)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.arange(3).float().unsqueeze(1).expand(3, 8))
        model.classifier.bias.copy_(torch.tensor([10.0, 11.0, 12.0]))
    model.save_pretrained(tmp_path)

    aligned = aligned_base(str(tmp_path))
    # v4 order is entailment, neutral, contradiction: the original rows 2, 1, 0.
    assert aligned.classifier.bias.tolist() == [12.0, 11.0, 10.0]
    assert aligned.classifier.weight[:, 0].tolist() == [2.0, 1.0, 0.0]
    assert aligned.config.id2label == {0: "entailment", 1: "neutral", 2: "contradiction"}
