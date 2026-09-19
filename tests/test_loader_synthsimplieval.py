"""Tests for the SynthSimpliEval loader.

There is nothing to load: no publicly retrievable file carries a human-judged score for any
SynthSimpliEval pair (see ``src/data/loaders/synthsimplieval.py`` and ``RAPPORT.md`` for the
investigation trail). These tests confirm the block is deliberate, informative, and stays
that way, and document the one real artifact found: score-less example sentences copied
verbatim from the paper's own LaTeX source.
"""

import json
from pathlib import Path

import pytest

from data.loaders.synthsimplieval import DataUnavailableError, load

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthsimplieval" / "paper_examples.json"


def test_load_always_raises_data_unavailable_error():
    with pytest.raises(DataUnavailableError):
        load()


def test_load_error_names_the_sources_that_were_checked():
    with pytest.raises(DataUnavailableError) as excinfo:
        load()
    message = str(excinfo.value)
    assert "github.com/jliu7350/text-simplification-benchmark" in message
    assert "arXiv:2504.09394" in message


def test_load_error_explains_the_llm_as_judge_conflict():
    with pytest.raises(DataUnavailableError, match="LLM-as-a-jury"):
        load()


def test_paper_examples_fixture_has_no_fabricated_scores():
    """The one fixture we ship documents real text with a real, honest absence of a label."""
    examples = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert examples["simplified_sample"]["label_raw"] is None
    assert len(examples["synthetic_complex_sentence_samples"]) == 3


def test_paper_examples_fixture_sentences_are_non_empty_real_text():
    examples = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for sample in examples["synthetic_complex_sentence_samples"]:
        assert sample["sentence"].strip()
    complex_sentence = examples["simplified_sample"]["complex"]
    assert complex_sentence.strip()
    for simplification in examples["simplified_sample"]["simplifications"].values():
        assert simplification.strip()
