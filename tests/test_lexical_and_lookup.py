"""Tests for the negation lookup, the lexical augmentation and the minimal-pair table.

The lookup is a baseline the paper compares against, so a lookup that misses a negation
would flatter the polarity head. The lexical augmentation must keep its two lexicon halves
disjoint, or the held-out antonyms of the minimal-pair table would not be held out.
"""

from __future__ import annotations

import random

import numpy as np

from data.build_lexical_augmentation import clean_edits, generate, split_half, test_words, train_tables
from diagnostics.minimal_pairs import ANTONYMS
from diagnostics.negation_lookup import flags
from figures_generator.minimal_pairs_table import right_side


class TestLookup:
    def test_flags_a_negation_in_one_sentence_only(self):
        assert flags(["A man is playing"], ["A man is not playing"]) == [True]

    def test_a_negation_in_both_sentences_is_not_flagged(self):
        """``Not all`` against ``some ... not``: both negated, no sign flip."""
        assert flags(["Not all people came"], ["Some people did not come"]) == [False]

    def test_contractions_count(self):
        assert flags(["The dog runs"], ["The dog doesn't run"]) == [True]


class TestLexiconHalves:
    def test_train_and_test_antonym_halves_are_disjoint(self):
        train, test = split_half(ANTONYMS)
        train_words = {word for pair in train for word in pair}
        test_half = {word for pair in test for word in pair}
        assert train_words.isdisjoint(test_half)

    def test_no_held_out_word_reaches_the_training_tables(self):
        antonyms, entailing = train_tables()
        caption_test = {word for pair in split_half(ANTONYMS)[1] for word in pair}
        assert caption_test.isdisjoint(antonyms)
        assert caption_test <= test_words()


class TestCleanEdits:
    def test_skips_compounds(self):
        assert clean_edits("Change-Up scored well", {"up": "down"}) == []

    def test_skips_proper_names_inside_the_sentence(self):
        assert clean_edits("Cheshire West and Chester", {"west": "east"}) == []

    def test_keeps_a_capitalised_first_word(self):
        got = clean_edits("Inside the house a man sleeps", {"inside": "outside"})
        assert got and got[0][0].startswith("Outside")

    def test_generate_labels_and_caps(self):
        rows = generate(["A man is inside", "A dog is inside"], {"inside": "outside"}, 2, random.Random(0), cap=1)
        assert len(rows) == 1 and rows[0]["polarity"] == 2.0


class TestRightSide:
    def test_counts_the_expected_side(self):
        signed = np.array([-5.0, 10.0, -1.0, 3.0])
        mask = np.array([True, True, True, False])
        assert right_side(signed, mask, "below") == 2 / 3 * 100
        assert right_side(signed, mask, "above") == 1 / 3 * 100

    def test_an_empty_column_is_nan_not_zero(self):
        assert np.isnan(right_side(np.array([1.0]), np.array([False]), "below"))
