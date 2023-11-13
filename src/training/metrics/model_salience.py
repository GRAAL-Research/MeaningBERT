import nltk
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import utils_masking


def unfold(sent_toks, make_tensor=True):
    unfolded = [w for sent in sent_toks for w in sent]
    if make_tensor:
        unfolded = torch.LongTensor(unfolded)
    return unfolded


class CoverageModel:
    def __init__(
        self,
        device="cuda",
        model_file=None,
    ):
        self.model_card = "roberta-base"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_card)

        self.eos_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.eos_token_id is None:
            self.eos_token_id = 0

        self.masking_model = utils_masking.NonStopMasker()
        self.masking_model.register_tokenizer(self.tokenizer)

        self.vocab_size = self.tokenizer.vocab_size
        self.device = device
        self.mask_id = 0

        self.model.half()

        self.model.to(self.device)
        if model_file is not None:
            self.reload_model(model_file)

    def reload_model(self, model_file):
        print(self.model.load_state_dict(torch.load(model_file), strict=False))

    def process_text(self, document):
        sentences = [" " + sent for sent in nltk.tokenize.sent_tokenize(document) if len(sent) > 0]
        (
            unmasked,
            masked,
            is_masked,
            effective_mask_ratio,
            all_masked_words_in_sentences,
        ) = self.masking_model.mask(sentences)
        return (
            unfold(unmasked),
            unfold(masked),
            unfold(is_masked),
            effective_mask_ratio,
            all_masked_words_in_sentences,
        )

    def build_io(self, targets, generateds):
        N = len(targets)

        input_ids, labels, is_masked, effective_mask_ratios = [], [], [], []
        gen_toks = []

        all_masked_words_in_sentences = []
        for target, generated in zip(targets, generateds):
            (
                unmasked,
                masked,
                is_ms,
                effective_mask_ratio,
                masked_words_in_sentences,
            ) = self.process_text(target)
            input_ids.append(masked)
            labels.append(unmasked)
            is_masked.append(is_ms)
            gen_toks.append(torch.LongTensor(self.tokenizer.encode(generated, add_special_tokens=False)))
            effective_mask_ratios.append(effective_mask_ratio)
            all_masked_words_in_sentences.append(masked_words_in_sentences)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        is_masked = torch.nn.utils.rnn.pad_sequence(is_masked, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
        input_ids = input_ids[:, :250]
        is_masked = is_masked[:, :250]
        labels = labels[:, :250]

        gen_toks = torch.nn.utils.rnn.pad_sequence(gen_toks, batch_first=True, padding_value=0)
        gen_toks = gen_toks[:, :250]
        gen_targets = torch.LongTensor([-1]).repeat(gen_toks.shape)

        seps = torch.LongTensor([self.eos_token_id]).repeat(N, 1)
        seps_targets = torch.LongTensor([-1]).repeat(seps.shape)

        input_ids = torch.cat((gen_toks, seps, input_ids), dim=1)
        labels = torch.cat((gen_targets, seps_targets, labels), dim=1)
        is_masked = torch.cat((torch.zeros_like(gen_toks), torch.zeros_like(seps), is_masked), dim=1)

        labels = labels.to(self.device)
        input_ids = input_ids.to(self.device)
        is_masked = is_masked.to(self.device)

        return (
            input_ids,
            is_masked,
            labels,
            effective_mask_ratios,
            all_masked_words_in_sentences,
        )

    def score(self, bodies, decodeds):
        score_func = self.score_soft
        unnorm_scores = score_func(bodies, decodeds)

        return unnorm_scores

    def score_soft(self, bodies, decodeds):
        (
            input_ids_w,
            is_masked_w,
            labels_w,
            effective_mask_ratios,
            all_masked_words_in_sentences,
        ) = self.build_io(bodies, decodeds)
        scores = self.score_soft_tokenized(input_ids_w, is_masked_w, labels_w)

        return {
            "scores": scores,
            "effective_mask_ratios": effective_mask_ratios,
            "original_sentence": bodies,
            "all_masked_words_in_sentences": all_masked_words_in_sentences,
        }

    def score_soft_tokenized(self, input_ids_w, is_masked_w, labels_w):
        self.model.eval()
        with torch.no_grad():
            outputs_w = self.model(input_ids_w)
            outputs_probs_w = torch.softmax(outputs_w["logits"], dim=2)
            max_probs, _ = outputs_probs_w.max(dim=2)

            relative_probs_w = (outputs_probs_w.permute(2, 0, 1) / max_probs).permute(1, 2, 0)

            batch_size, seq_len = is_masked_w.shape
            t_range = torch.arange(seq_len)

            scores = []
            for seq_rel_probs, seq_labels, seq_is_masked in zip(relative_probs_w, labels_w, is_masked_w):
                selected_probs = (seq_rel_probs[t_range, seq_labels]) * seq_is_masked
                soft_score = torch.sum(selected_probs) / (torch.sum(seq_is_masked) + 0.1)
                scores.append(soft_score.item())

        return scores
