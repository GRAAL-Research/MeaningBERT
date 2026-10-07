"""Tests for the minimal-pair builder (``src/diagnostics/minimal_pairs.py``).

A minimal pair that differs by two words, or an antonym that fires inside a longer word,
would credit the polarity head with a distinction it was never shown. These tests pin the
one-word guarantee and the edits the expected signs rest on.
"""

from __future__ import annotations

import random

from diagnostics.minimal_pairs import build_pairs, chain, lexicon, negate, substitute


def _words(sentence: str) -> list[str]:
    return sentence.split()


class TestNegate:
    def test_inserts_not_after_the_first_copula(self):
        assert negate("A man is playing a guitar") == "A man is not playing a guitar"

    def test_leaves_an_already_negated_sentence_out(self):
        """A double negation would turn the expected contradiction back into agreement."""
        assert negate("There is no man playing a guitar") is None

    def test_a_sentence_without_copula_gives_none(self):
        assert negate("Two dogs run in the park") is None


class TestSubstitute:
    def test_whole_words_only(self):
        """``man`` must not fire inside ``woman``."""
        got = substitute("A woman is walking", lexicon("cohyponym"))
        olds = [old for _edited, old, _new in got]
        assert "woman" in olds and "man" not in olds
        assert all("womwoman" not in edited and "wowoman" not in edited for edited, _o, _n in got)

    def test_each_edit_changes_exactly_one_word(self):
        source = "A little girl is smiling and running outside"
        for edited, _old, _new in substitute(source, lexicon("antonym")):
            diff = [a for a, b in zip(_words(source), _words(edited)) if a != b]
            assert len(diff) == 1

    def test_antonyms_are_read_both_ways(self):
        table = lexicon("antonym")
        assert table["inside"] == "outside" and table["outside"] == "inside"

    def test_capitalisation_follows_the_replaced_word(self):
        got = substitute("Dog is running", lexicon("hypernym"))
        assert got[0][0] == "Animal is running"


class TestChain:
    def test_step_k_differs_from_the_source_by_k_words(self):
        source = "A little girl is smiling and running outside on the grass"
        versions = chain(source, random.Random(0), steps=4)
        assert versions is not None and len(versions) == 4
        for step, edited in enumerate(versions, start=1):
            diff = [a for a, b in zip(_words(source), _words(edited)) if a != b]
            assert len(diff) == step

    def test_too_few_editable_words_gives_none(self):
        assert chain("A person sits", random.Random(0), steps=4) is None


class TestBuild:
    def test_kinds_are_capped_and_chains_are_kept_whole(self):
        sentences = ["A man is playing a guitar outside", "A dog is running inside on the wet grass"]
        pairs = build_pairs(sentences, per_kind=1)
        kinds = [pair["kind"] for pair in pairs if pair["kind"] != "chain"]
        assert all(kinds.count(kind) <= 1 for kind in set(kinds))
        assert "negation" in kinds
