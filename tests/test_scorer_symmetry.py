"""The scorer scores one direction, and the asymmetry it leaves is a measured result.

``meaning(A, B)`` and ``meaning(B, A)`` ask the same question, so they should return the
same number. The corpus teaches that through mirrored training pairs, and it works: on the
published configuration the two directions differ by 1.48 points on average, against 7.79
without them.

What the scorer does NOT do is average the two directions at inference. That would make the
property exact, at double the cost, by patching outside the model what the model should
hold on its own, and it would hide the residual violation instead of reporting it. These
tests pin that decision: one call, one direction, one forward pass.

The stubs below replace the tokenizer and the model, never ``score()`` itself. Stubbing the
method under test would leave these tests passing whatever the scorer does.
"""

import contextlib

import pytest

from meaningbert import scorer as scorer_module
from meaningbert.scorer import MeaningBERTScorer


class _Recording:
    """A tokenizer and model pair that records the calls and is blatantly order-dependent.

    The fake model returns the length of the FIRST sentence of each pair, so any averaging
    of the two directions would show up immediately as a different number.
    """

    def __init__(self) -> None:
        self.batches: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def tokenizer(self, batch_a, batch_b, **_kwargs):
        self.batches.append((tuple(batch_a), tuple(batch_b)))
        return self

    def to(self, _device):
        # score() does ``model(**encoded)``, so this has to be a mapping, not the recorder.
        return {"encoded": True}

    def __call__(self, **_kwargs):
        lengths = [float(len(a)) for a in self.batches[-1][0]]
        return type("Output", (), {"logits": lengths})()


class _NoTorch:
    """Just enough of torch for ``score()``: the no_grad context manager.

    The head arithmetic is stubbed out separately, so this test needs no tensors and runs
    on a machine without torch installed, which is where the rest of the suite runs.
    """

    @staticmethod
    def no_grad():
        return contextlib.nullcontext()


@pytest.fixture(autouse=True)
def _passthrough_head(monkeypatch):
    """Replace the 0-100 mapping by identity.

    What is under test is the direction and the batching, not the head arithmetic, which
    has its own suite. Stubbing it keeps this file free of torch.
    """
    monkeypatch.setattr(scorer_module, "to_percent", lambda logits, head: _Scores(logits))


class _Scores(list):
    """A list that answers the two calls ``score()`` makes on a tensor."""

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


def _scorer(batch_size: int = 32) -> tuple[MeaningBERTScorer, _Recording]:
    """A scorer wired to the fake pieces, without loading any weights."""
    recorder = _Recording()
    scorer = MeaningBERTScorer.__new__(MeaningBERTScorer)
    scorer.batch_size = batch_size
    scorer.device = "cpu"
    scorer.max_length = 256
    scorer.head = "linear"  # The head whose logit IS the score, so the stub passes through.
    scorer._tokenizer = recorder.tokenizer  # noqa: SLF001
    scorer._model = recorder  # noqa: SLF001
    scorer._torch = _NoTorch  # noqa: SLF001
    return scorer, recorder


def test_the_score_depends_on_the_order_because_only_one_direction_is_run():
    """The asymmetry is left visible on purpose; it is the article's diagnostic."""
    scorer, _ = _scorer()
    assert scorer.score(["aaa"], ["b"]) == [3.0]
    assert scorer.score(["b"], ["aaa"]) == [1.0]


def test_the_model_never_sees_the_mirrored_pair():
    """The decisive test: no hidden second pass with the arguments swapped."""
    scorer, recorder = _scorer()
    scorer.score(["aaa", "bb"], ["c", "d"])
    assert recorder.batches == [(("aaa", "bb"), ("c", "d"))]


def test_a_long_input_is_batched_but_still_one_direction():
    """Batching must not sneak the mirrored direction in as an extra batch."""
    scorer, recorder = _scorer(batch_size=2)
    scorer.score(["a", "bb", "ccc", "dddd", "e"], ["z"] * 5)
    assert [b[0] for b in recorder.batches] == [("a", "bb"), ("ccc", "dddd"), ("e",)]


def test_the_scorer_exposes_no_symmetry_switch():
    """The published API has one behaviour, so there is no flag to get it wrong with."""
    assert not hasattr(MeaningBERTScorer, "symmetric")
    assert "symmetric" not in getattr(MeaningBERTScorer, "__dataclass_fields__", {})


def test_mismatched_lengths_are_refused_before_any_forward_pass():
    scorer, recorder = _scorer()
    with pytest.raises(ValueError):
        scorer.score(["a", "b"], ["c"])
    assert recorder.batches == []


def test_an_empty_input_returns_no_score_and_runs_nothing():
    scorer, recorder = _scorer()
    assert scorer.score([], []) == []
    assert recorder.batches == []
