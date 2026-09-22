"""The scorer must return the same number whatever the order of the two sentences.

This is not a quality target, it is an invariant. ``meaning(A, B)`` and ``meaning(B, A)``
ask the same question: how much of the meaning is shared. The position of a sentence in the
call carries no semantic information, so a different answer is a defect.

Measurement is what makes these tests necessary. On 1652 test pairs, the published v1 model
disagrees with itself by 6.12 points on average and by more than 10 points on 20.8 percent
of them; a v2 model trained without mirrored pairs is worse still, at 7.79. Training does
not deliver the property, so the API enforces it.
"""

import pytest

from meaningbert.scorer import MeaningBERTScorer


class _Asymmetric(MeaningBERTScorer):
    """A scorer whose underlying model is blatantly order-dependent.

    Built by hand rather than loaded, so the test needs neither weights nor a GPU. The
    stub returns the length of the first argument, which makes the asymmetry both extreme
    and trivially predictable.
    """

    def __init__(self, symmetric: bool = True) -> None:  # pylint: disable=super-init-not-called
        self.symmetric = symmetric
        self.batch_size = 32

    def _score_one_way(self, documents, simplifications):
        return [float(len(d)) for d in documents]


def test_the_stub_really_is_asymmetric():
    """Guard the guard: a symmetric stub would make every other test pass for free."""
    raw = _Asymmetric(symmetric=False)
    assert raw.score(["aaa"], ["b"]) == [3.0]
    assert raw.score(["b"], ["aaa"]) == [1.0]


def test_swapping_the_arguments_does_not_change_the_score():
    scorer = _Asymmetric()
    assert scorer.score(["aaa"], ["b"]) == scorer.score(["b"], ["aaa"])


def test_the_symmetric_score_is_the_mean_of_the_two_directions():
    """Not just equal, equal to the right thing.

    The mean is the only symmetrisation that leaves an already-symmetric model untouched;
    min or max would shift every score of a model that has the property.
    """
    assert _Asymmetric().score(["aaa"], ["b"]) == [2.0]


def test_symmetry_can_be_turned_off_to_measure_the_violation():
    """The article reports the raw violation as a diagnostic, so it must stay reachable."""
    assert _Asymmetric(symmetric=False).score(["aaa"], ["b"]) == [3.0]


def test_mismatched_lengths_are_refused_before_any_forward_pass():
    with pytest.raises(ValueError):
        _Asymmetric().score(["a", "b"], ["c"])


def test_an_empty_input_returns_no_score_rather_than_failing():
    assert _Asymmetric().score([], []) == []
