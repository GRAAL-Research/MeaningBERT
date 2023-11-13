import nltk
import numpy as np


class NonStopMasker:
    def __init__(self):
        # Masks everything but stop words
        self.model_tokenizer = None
        self.stop_ws = set(nltk.corpus.stopwords.words("english"))

    def register_tokenizer(self, tokenizer):
        self.model_tokenizer = tokenizer

    @staticmethod
    def compute_effective_mask_ratio(is_masked):
        return np.mean([np.mean(is_m) for is_m in is_masked])

    def mask(self, sentences):
        unmasked, masked, is_masked = [], [], []

        masked_words_in_sentences = []
        for sentence in sentences:
            masked_words_in_sentence = []
            ums, ms, ims = [], [], []
            words = nltk.tokenize.word_tokenize(sentence)
            even = 0
            for w in words:
                toks = self.model_tokenizer.encode(" " + w, add_special_tokens=False)
                ums += toks
                even += 1
                if w.lower() not in self.stop_ws and even % 2 == 0:
                    ms += [0] * len(toks)
                    ims += [1] * len(toks)
                    masked_words_in_sentence.append(w)
                else:
                    ms += toks
                    ims += [0] * len(toks)
            masked_words_in_sentences.extend(masked_words_in_sentence)
            unmasked.append(ums)
            masked.append(ms)
            is_masked.append(ims)
        return (
            unmasked,
            masked,
            is_masked,
            self.compute_effective_mask_ratio(is_masked),
            masked_words_in_sentences,
        )
