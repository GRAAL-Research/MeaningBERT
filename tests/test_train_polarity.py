"""Tests for the polarity head's metrics (``src/training/train_polarity.py``).

Only the pure reporting half is tested here; the training loop is a thin wrapper over the
Trainer and testing it would mean testing transformers. What is worth testing is what the
experiment will be read from: the confusion matrix, whose orientation decides whether a
result says "the head calls contradictions neutral" or the opposite, and macro-F1, which
is the only number that notices a head refusing to predict a minority class.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from data.schema import POLARITY_CLASSES
from training.train_polarity import (
    CLASS_NAMES,
    compute_metrics,
    confusion,
    degenerate,
    divergent_log,
    encode_pair,
    needs_pair_template,
    per_class_f1,
    summarise,
)

ENTAIL = POLARITY_CLASSES["entailment"]
NEUTRAL = POLARITY_CLASSES["neutral"]
CONTRA = POLARITY_CLASSES["contradiction"]


def _arrays(pairs):
    """From (true, predicted) pairs to the two arrays the reporters take."""
    return np.array([p for _, p in pairs]), np.array([t for t, _ in pairs])


# --- class order ---------------------------------------------------------------------


def test_the_class_names_follow_the_schema_indices():
    # Every matrix row and column is read by position, so a name list out of step with the
    # indices would relabel the whole report without changing a number.
    assert CLASS_NAMES == ("entailment", "neutral", "contradiction")
    assert [POLARITY_CLASSES[name] for name in CLASS_NAMES] == [0, 1, 2]


# --- the confusion matrix ------------------------------------------------------------


def test_a_perfect_classifier_fills_only_the_diagonal():
    predictions, labels = _arrays([(ENTAIL, ENTAIL), (NEUTRAL, NEUTRAL), (CONTRA, CONTRA)])
    assert confusion(predictions, labels) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_the_matrix_reads_row_true_column_predicted():
    # The orientation IS the finding. Calling a contradiction neutral costs the sign;
    # calling a neutral pair a contradiction invents one. A transposed matrix reports the
    # opposite failure with the same numbers.
    predictions, labels = _arrays([(CONTRA, NEUTRAL)])
    matrix = confusion(predictions, labels)
    assert matrix[CONTRA][NEUTRAL] == 1
    assert matrix[NEUTRAL][CONTRA] == 0


def test_every_prediction_lands_somewhere_in_the_matrix():
    pairs = [(ENTAIL, NEUTRAL), (NEUTRAL, NEUTRAL), (CONTRA, ENTAIL), (CONTRA, CONTRA)]
    predictions, labels = _arrays(pairs)
    assert sum(sum(row) for row in confusion(predictions, labels)) == len(pairs)


# --- F1 ------------------------------------------------------------------------------


def test_a_class_never_predicted_scores_zero_not_undefined():
    # The failure macro-F1 exists to catch: VitaminC's test split is 17 % neutral, so a
    # head that never predicts neutral still reaches a respectable accuracy.
    pairs = [(NEUTRAL, ENTAIL)] * 3 + [(ENTAIL, ENTAIL)] * 7
    predictions, labels = _arrays(pairs)
    got = summarise(predictions, labels)
    assert got["f1"]["neutral"] == 0.0
    assert got["accuracy"] == pytest.approx(0.7)
    assert got["macro_f1"] < got["accuracy"]


def test_macro_f1_is_the_unweighted_mean_over_the_three_classes():
    matrix = [[2, 0, 0], [0, 1, 1], [0, 0, 2]]
    scores = per_class_f1(matrix)
    assert scores["macro"] == pytest.approx(sum(scores[name] for name in CLASS_NAMES) / 3)
    assert scores["entailment"] == pytest.approx(1.0)


def test_f1_balances_precision_against_recall():
    # Predicted contradiction 4 times, right twice; 2 of 3 true contradictions found.
    # precision 0.5, recall 2/3, F1 = 2*0.5*(2/3)/(0.5+2/3) = 4/7.
    matrix = [[0, 0, 2], [0, 0, 0], [1, 0, 2]]
    assert per_class_f1(matrix)["contradiction"] == pytest.approx(4 / 7)


def test_a_perfect_classifier_scores_one_everywhere():
    got = summarise(*_arrays([(ENTAIL, ENTAIL), (NEUTRAL, NEUTRAL), (CONTRA, CONTRA)]))
    assert got["accuracy"] == pytest.approx(1.0)
    assert got["macro_f1"] == pytest.approx(1.0)


def test_an_empty_evaluation_reports_nan_rather_than_a_perfect_score():
    # Zero correct out of zero must not read as 100 %, and it must not crash a run that
    # has already paid for its training.
    got = summarise(np.array([]), np.array([]))
    assert math.isnan(got["accuracy"])
    assert got["n"] == 0


# --- the Trainer hook ----------------------------------------------------------------


def test_the_trainer_hook_turns_logits_into_the_two_scalars_it_tracks():
    # The Trainer hands logits, not classes. Forgetting the argmax would compare raw
    # scores to class indices and report a plausible, meaningless accuracy.
    logits = np.array([[9.0, 0.0, 0.0], [0.0, 0.0, 9.0], [0.0, 9.0, 0.0]])
    got = compute_metrics((logits, np.array([ENTAIL, CONTRA, NEUTRAL])))
    assert got == {"accuracy": pytest.approx(1.0), "macro_f1": pytest.approx(1.0)}


def test_the_trainer_hook_notices_a_wrong_prediction():
    logits = np.array([[9.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    got = compute_metrics((logits, np.array([ENTAIL, CONTRA])))
    assert got["accuracy"] == pytest.approx(0.5)


# --- reusing a pretrained NLI head, and refusing to reuse it wrong --------------------

from training.train_polarity import head_reuse_plan  # noqa: E402


def test_a_checkpoint_already_in_our_order_is_reused_untouched():
    # MoritzLaurer's NLI models publish exactly this order, so the head is reused as is and
    # the run starts from a model that already does the task.
    reusable, permutation = head_reuse_plan({0: "entailment", 1: "neutral", 2: "contradiction"})
    assert reusable
    assert permutation == [0, 1, 2]


def test_a_reversed_checkpoint_is_reused_but_permuted():
    # roberta-large-mnli publishes the reverse. Three labels either way, so nothing
    # mismatches and the head is silently kept: this is the same failure as the swapped
    # SICK encoding, wrong without ever crashing.
    reusable, permutation = head_reuse_plan({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"})
    assert reusable
    assert permutation == [2, 1, 0]


def test_the_permutation_maps_the_checkpoint_rows_into_our_order():
    _, permutation = head_reuse_plan({0: "contradiction", 1: "neutral", 2: "entailment"})
    reordered = [["row_contra"], ["row_neutral"], ["row_entail"]]
    got = [reordered[index] for index in permutation]
    assert got == [["row_entail"], ["row_neutral"], ["row_contra"]]


def test_a_plain_encoder_has_nothing_to_reuse():
    # bert-base-uncased and microsoft/deberta-v3-large publish LABEL_0 / LABEL_1: not an
    # inference label space, so the head is replaced and nothing is warned about.
    assert head_reuse_plan({0: "LABEL_0", 1: "LABEL_1"}) == (False, None)
    assert head_reuse_plan({0: "LABEL_0"}) == (False, None)
    assert head_reuse_plan(None) == (False, None)


def test_a_three_class_head_that_is_not_an_inference_space_is_left_alone():
    assert head_reuse_plan({0: "positive", 1: "negative", 2: "mixed"}) == (False, None)


def test_a_duplicated_class_name_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="not a permutation"):
        head_reuse_plan({0: "entailment", 1: "entailment", 2: "contradiction"})


def test_the_case_of_the_published_labels_does_not_matter():
    assert head_reuse_plan({0: "ENTAILMENT", 1: "Neutral", 2: " contradiction "})[1] == [0, 1, 2]


# --- finding the head, which is not the same object across families ------------------

from training.train_polarity import output_layer  # noqa: E402


class _Linear:
    def __init__(self, rows=3, cols=8):
        self.weight = type("W", (), {"dim": lambda self: 2, "shape": (rows, cols)})()


class _DebertaLike:
    """classifier IS the linear layer, as in DebertaV2ForSequenceClassification."""

    def __init__(self):
        self.classifier = _Linear()


class _RobertaLike:
    """classifier is a head module whose logits come out of out_proj."""

    class _Head:
        def __init__(self):
            self.out_proj = _Linear()

    def __init__(self):
        self.classifier = self._Head()


class _Opaque:
    class _Head:
        pass

    def __init__(self):
        self.classifier = self._Head()


def test_a_deberta_head_is_its_own_linear_layer():
    model = _DebertaLike()
    assert output_layer(model) is model.classifier


def test_a_roberta_head_hides_its_linear_layer_in_out_proj():
    # Reaching for classifier.weight here raises, and roberta-large-mnli is precisely the
    # checkpoint whose rows need permuting.
    model = _RobertaLike()
    assert output_layer(model) is model.classifier.out_proj


def test_an_unrecognised_head_raises_instead_of_permuting_the_wrong_tensor():
    with pytest.raises(AttributeError, match="no output linear layer"):
        output_layer(_Opaque())


class _StubTokenizer:
    def __init__(self, pad=None, eos="</s>", eos_id=2):
        self.pad_token = pad
        self.pad_token_id = None if pad is None else 0
        self.eos_token = eos
        self.eos_token_id = eos_id

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "pad_token" and value is not None and getattr(self, "eos_token", None) == value:
            super().__setattr__("pad_token_id", self.eos_token_id)


class _StubConfig:
    def __init__(self, pad_token_id=None):
        self.pad_token_id = pad_token_id


class _StubModel:
    def __init__(self, pad_token_id=None):
        self.config = _StubConfig(pad_token_id)


class TestAlignPadding:
    """A decoder-only checkpoint has no pad token, and a batched classifier needs one."""

    def test_a_decoder_only_checkpoint_borrows_its_end_of_sequence_token(self):
        from training.train_polarity import align_padding

        tokenizer, model = _StubTokenizer(pad=None), _StubModel()

        assert align_padding(tokenizer, model) is True
        assert tokenizer.pad_token == tokenizer.eos_token
        assert model.config.pad_token_id == tokenizer.eos_token_id

    def test_an_encoder_that_already_pads_is_left_alone(self):
        """Touching a checkpoint that ships a pad token would change 140 published runs."""
        from training.train_polarity import align_padding

        tokenizer, model = _StubTokenizer(pad="[PAD]"), _StubModel(pad_token_id=0)

        assert align_padding(tokenizer, model) is False
        assert tokenizer.pad_token == "[PAD]"

    def test_a_model_whose_config_forgot_the_id_is_repaired(self):
        """An unset pad_token_id pools over padding as if it were text, and says nothing."""
        from training.train_polarity import align_padding

        tokenizer, model = _StubTokenizer(pad="[PAD]"), _StubModel(pad_token_id=None)

        assert align_padding(tokenizer, model) is True
        assert model.config.pad_token_id == 0

    def test_a_checkpoint_with_neither_token_raises_rather_than_padding_with_zero(self):
        from training.train_polarity import align_padding

        tokenizer = _StubTokenizer(pad=None, eos=None, eos_id=None)

        with pytest.raises(ValueError, match="end-of-sequence"):
            align_padding(tokenizer, _StubModel())


class TestDegenerate:
    """La garde qui refuse d'ecrire un resultat effondre."""

    @staticmethod
    def _resume(matrix, macro_f1=0.5):
        return {"confusion": matrix, "macro_f1": macro_f1}

    def test_une_seule_classe_predite_est_signalee(self):
        # Les neuf cellules d'AUC du 4 octobre 2026 : tout dans entailment.
        matrix = [[4000, 0, 0], [4000, 0, 0], [4000, 0, 0]]
        fault = degenerate(self._resume(matrix, macro_f1=1 / 6))
        assert fault is not None
        assert "entailment" in fault

    def test_aucune_prediction_est_signalee(self):
        fault = degenerate(self._resume([[0, 0, 0]] * 3))
        assert fault is not None
        assert "aucune" in fault

    def test_macro_f1_non_fini_est_signale(self):
        matrix = [[300, 50, 50], [40, 310, 50], [30, 60, 310]]
        fault = degenerate(self._resume(matrix, macro_f1=float("nan")))
        assert fault == "macro-F1 non fini"

    def test_deux_classes_predites_passent(self):
        # Un modele faible mais vivant n'est pas refuse : deux classes suffisent.
        matrix = [[300, 100, 0], [200, 200, 0], [150, 250, 0]]
        assert degenerate(self._resume(matrix)) is None

    def test_matrice_saine_passe(self):
        matrix = [[380, 10, 10], [15, 370, 15], [20, 20, 360]]
        assert degenerate(self._resume(matrix, macro_f1=0.92)) is None

    def test_resume_reel_est_accepte(self):
        # Le chemin complet : summarise produit le dict que degenerate inspecte.
        predictions = np.array([0, 1, 2, 0, 1, 2])
        labels = np.array([0, 1, 2, 0, 2, 1])
        assert degenerate(summarise(predictions, labels)) is None

    def test_resume_reel_effondre_est_refuse(self):
        predictions = np.zeros(6, dtype=int)
        labels = np.array([0, 1, 2, 0, 1, 2])
        assert degenerate(summarise(predictions, labels)) is not None


class TestDivergentLog:
    """La garde qui coupe des la premiere ligne non finie."""

    def test_grad_norm_nan_est_signale(self):
        # La ligne exacte des cellules d'AUC : perte masquee a zero, gradient NaN.
        fault = divergent_log({"loss": 0.0, "grad_norm": float("nan"), "epoch": 0.042})
        assert fault is not None
        assert "grad_norm" in fault

    def test_perte_infinie_est_signalee(self):
        assert "loss" in divergent_log({"loss": float("inf")})

    def test_perte_enorme_mais_finie_passe(self):
        # 3e12 est absurde mais fini : c'est le gradient qui tranche, pas l'ampleur.
        assert divergent_log({"loss": 3.086e12, "grad_norm": 2.0}) is None

    def test_ligne_saine_passe(self):
        assert divergent_log({"loss": 0.41, "grad_norm": 1.2, "learning_rate": 9e-6}) is None

    def test_ligne_sans_perte_passe(self):
        # Les lignes d'evaluation n'ont pas ces cles ; elles ne doivent pas lever.
        assert divergent_log({"eval_accuracy": 0.9, "epoch": 1.0}) is None

    def test_logs_absents_passent(self):
        assert divergent_log(None) is None

    def test_valeur_non_numerique_est_ignoree(self):
        assert divergent_log({"loss": "indisponible"}) is None


class _Tokeniseur:
    """Un tokeniseur reduit a ce que les deux fonctions regardent."""

    def __init__(self, sep_token_id):
        self.sep_token_id = sep_token_id


class TestEncodePair:
    """La frontiere entre les deux phrases doit exister pour tout modele."""

    def test_encodeur_garde_la_paire_telle_quelle(self):
        # DeBERTa produit [CLS] a [SEP] b [SEP] : rien a reformuler.
        tok = _Tokeniseur(sep_token_id=2)
        first, second = encode_pair(tok, ["un chien court"], ["un animal court"])
        assert first == ["un chien court"]
        assert second == ["un animal court"]

    def test_decodeur_recoit_un_gabarit(self):
        # SmolLM2 n'a pas de separateur : sans gabarit les deux phrases sont collees.
        tok = _Tokeniseur(sep_token_id=None)
        first, second = encode_pair(tok, ["un chien court"], ["un animal court"])
        assert second is None
        assert first == ["premise: un chien court\nhypothesis: un animal court"]

    def test_le_gabarit_separe_vraiment_les_deux_phrases(self):
        # Le defaut du 4 octobre 2026 : la concatenation nue, sans frontiere.
        tok = _Tokeniseur(sep_token_id=None)
        first, _ = encode_pair(tok, ["un chien court"], ["un animal court"])
        assert "un chien courtun animal court" not in first[0]
        assert first[0].index("un chien court") < first[0].index("un animal court")

    def test_le_gabarit_porte_sur_tout_le_lot(self):
        tok = _Tokeniseur(sep_token_id=None)
        first, second = encode_pair(tok, ["a", "b", "c"], ["x", "y", "z"])
        assert len(first) == 3
        assert second is None
        assert first[2] == "premise: c\nhypothesis: z"

    def test_lot_vide(self):
        tok = _Tokeniseur(sep_token_id=None)
        first, second = encode_pair(tok, [], [])
        assert first == []
        assert second is None

    def test_detection_du_besoin(self):
        assert needs_pair_template(_Tokeniseur(sep_token_id=None)) is True
        assert needs_pair_template(_Tokeniseur(sep_token_id=2)) is False
        # L'identifiant zero est un vrai identifiant, pas une absence.
        assert needs_pair_template(_Tokeniseur(sep_token_id=0)) is False
