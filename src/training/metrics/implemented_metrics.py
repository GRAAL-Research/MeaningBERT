from typing import List

import numpy as np
from evaluate import load


class SentenceBertWrapper:
    def __init__(self, model_name: str = "sentence-t5-xxl"):
        # Imported here rather than at module level: sentence-transformers pulls a large
        # dependency tree that only this one comparison baseline needs. At module level it
        # made the whole module unimportable without it, which took sigmoid() -- a pure
        # four-line function -- down with it, and with it the only test file covering it.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def compute(self, prediction: str, reference: str) -> float:
        from sentence_transformers import util

        sentence_embedding_prediction = self.model.encode(prediction, convert_to_tensor=True)
        sentence_embedding_reference = self.model.encode(reference, convert_to_tensor=True)

        return util.cos_sim(sentence_embedding_reference, sentence_embedding_prediction)


class IBLEU:
    def __init__(self, alpha: float = 0.9):
        self.bleu = load("sacrebleu")
        self.alpha = alpha

    def compute(self, source: List[str], prediction: List[str], reference: List[str]) -> float:
        return (
            self.alpha * self.bleu.compute(predictions=prediction, references=[reference])["score"]
            - (1 - self.alpha) * self.bleu.compute(predictions=prediction, references=[source])["score"]
        )


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class FKBLEU:
    def __init__(self, alpha: float = 0.9):
        self.ibleu = IBLEU(alpha=alpha)

    def compute(self, source: List[str], prediction: List[str], reference: List[str]) -> float:
        # Imported here for the same reason as sentence-transformers above: an optional
        # comparison baseline must not decide whether this module can be imported.
        from textstat import textstat

        # I: input original -> source
        # R: reference -> reference (ground truth)
        # O: Output simplification -> prediction
        fkgl_prediction = textstat.flesch_kincaid_grade(prediction[0])
        fkgl_original = textstat.flesch_kincaid_grade(source[0])

        fkgl_diff = sigmoid(fkgl_prediction - fkgl_original)

        return self.ibleu.compute(source=source, reference=reference, prediction=prediction) * fkgl_diff
