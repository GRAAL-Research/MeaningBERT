"""Public inference surface of MeaningBERT, used by the HuggingFace metric."""

from meaningbert.scorer import MeaningBERTScorer, score_pairs

__all__ = ["MeaningBERTScorer", "score_pairs"]
