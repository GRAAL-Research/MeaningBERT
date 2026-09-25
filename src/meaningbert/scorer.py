"""Turn a MeaningBERT checkpoint into 0-100 meaning-preservation scores.

Why this exists. A v2 checkpoint is trained with a bounded output head, but the head is
applied *outside* the model: ``AutoModelForSequenceClassification`` emits a raw logit, and
`100 * sigmoid(logit)` is computed in the training loop. So the published usage from the
v1 model card::

    scores = scorer(**tokenize_text)
    print(scores.logits.tolist())

would return values like ``-2.3`` for a v2 checkpoint, silently, with no error. The
transformation has to travel with the weights, and the config is the only thing that does.
Training writes ``config.meaningbert_output_head``; this module reads it back and applies
the matching transformation.

A v1 checkpoint has no such field. It was trained with an unbounded linear head whose
logits *are* the score, so the absence of the field is itself the right answer and the
default is ``linear``.

Usage::

    from meaningbert import MeaningBERTScorer

    scorer = MeaningBERTScorer("davebulaval/MeaningBERT")
    scorer.score(["He wanted to make them pay."], ["He wanted revenge."])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

#: Default when a checkpoint's config says nothing: the v1 behaviour.
DEFAULT_HEAD: str = "linear"
SCORE_MIN: float = 0.0
SCORE_MAX: float = 100.0

#: Longest pair the model can attend to. DeBERTa-v3's tokenizer reports 1e30, so the bound
#: has to come from the config, never from ``truncation=True`` alone.
FALLBACK_MAX_LENGTH: int = 512


def head_of(config: Any) -> str:
    """Read the output head a checkpoint was trained with.

    Args:
        config: A loaded ``PretrainedConfig``.

    Returns:
        The head name, defaulting to ``linear`` for checkpoints predating the field.
    """
    return str(getattr(config, "meaningbert_output_head", DEFAULT_HEAD) or DEFAULT_HEAD)


def to_percent(logits: Any, head: str) -> Any:
    """Map raw model output to the 0-100 scale for *head*, then clip.

    Args:
        logits: Model output, a torch tensor of shape ``(n,)`` or ``(n, 1)``.
        head: ``linear``, ``sigmoid``, ``normalized`` or ``clamped``.

    Returns:
        A tensor of scores in ``[0, 100]``.

    Raises:
        ValueError: On an unknown head, rather than guessing and returning wrong numbers.
    """
    import torch  # local import: this module is importable without torch

    values = torch.as_tensor(logits).squeeze(-1) if getattr(logits, "ndim", 1) > 1 else torch.as_tensor(logits)
    if head == "sigmoid":
        values = values.sigmoid() * SCORE_MAX
    elif head in ("normalized", "clamped"):
        values = values * SCORE_MAX
    elif head != "linear":
        raise ValueError(f"unknown output head {head!r}; a wrong guess here returns plausible wrong scores")
    return values.clamp(SCORE_MIN, SCORE_MAX)


@dataclass
class MeaningBERTScorer:
    """Score meaning preservation between pairs of sentences.

    Loads the checkpoint once and applies whatever output head it was trained with, so the
    caller never has to know which one that was.
    """

    checkpoint: str = "davebulaval/MeaningBERT"
    #: Subfolder inside the repository, because the v2 checkpoints live there.
    #:
    #: The repository root is still the v1 model, so ``MeaningBERTScorer("davebulaval/
    #: MeaningBERT")`` keeps returning exactly the v1 scores and nothing already written
    #: changes behaviour. The v2 checkpoints are ``subfolder="large"`` and, later,
    #: ``subfolder="base"``.
    subfolder: Optional[str] = None
    device: Optional[str] = None
    batch_size: int = 32

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        where = {"subfolder": self.subfolder} if self.subfolder else {}
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, **where)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self.checkpoint, **where).to(self.device).eval()
        )
        self.head = head_of(self._model.config)
        self.max_length = min(
            getattr(self._model.config, "max_position_embeddings", FALLBACK_MAX_LENGTH) or FALLBACK_MAX_LENGTH,
            FALLBACK_MAX_LENGTH,
        )

    def score(self, documents: Sequence[str], simplifications: Sequence[str]) -> list[float]:
        """Score each ``(document, simplification)`` pair from 0 to 100.

        Args:
            documents: Source sentences.
            simplifications: Candidate sentences, aligned with *documents*.

        Returns:
            One score per pair, higher meaning better preserved.

        Raises:
            ValueError: If the two sequences have different lengths, which would silently
                score the wrong pairs against each other.
        """
        if len(documents) != len(simplifications):
            raise ValueError(f"got {len(documents)} documents and {len(simplifications)} simplifications")
        if not documents:
            return []

        # One direction, one forward pass. The metric is MEANT to be symmetric, and the
        # mirrored pairs in the training corpus are what teaches it: on the published
        # configuration the two directions differ by 1.48 points on average, against 7.79
        # without them. Averaging the two directions here would make the property exact,
        # but at double the inference cost and by patching outside the model what the
        # model should hold on its own. The residual violation is a result to report, not
        # something for the scorer to hide. ``src/diagnostics/evaluate_checkpoint.py``
        # measures it.
        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch_a = list(documents[start : start + self.batch_size])
            batch_b = list(simplifications[start : start + self.batch_size])
            encoded = self._tokenizer(
                batch_a, batch_b, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
            ).to(self.device)
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
            scores.extend(to_percent(logits, self.head).cpu().tolist())
        return scores


def score_pairs(
    documents: Sequence[str],
    simplifications: Sequence[str],
    checkpoint: str = "davebulaval/MeaningBERT",
    **kwargs: Any,
) -> list[float]:
    """One-shot convenience wrapper around :class:`MeaningBERTScorer`."""
    return MeaningBERTScorer(checkpoint=checkpoint, **kwargs).score(documents, simplifications)
