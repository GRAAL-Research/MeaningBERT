"""Tests for the French translation of the polarity corpus (``translate_polarity_corpus.py``).

The translated corpora feed two of the three v4 regimes. What can go wrong silently is a
label separated from its pair, a split boundary lost in the mixing, or a queue starting on
a corpus still being written; each is pinned here without loading a translation model.
"""

from __future__ import annotations

import os

from datasets import Dataset, DatasetDict

from data.translate_polarity_corpus import mixed, translated, unique_sentences, write


def _corpus() -> DatasetDict:
    rows = {"original": ["A dog runs.", "A cat sleeps."], "simplification": ["A dog does not run.", "A dog runs."]}
    labels = {"polarity": [2, 0], "corpus": ["sick", "lexical"]}
    return DatasetDict({"train": Dataset.from_dict({**rows, **labels}), "test": Dataset.from_dict({**rows, **labels})})


TABLE = {
    "A dog runs.": "Un chien court.",
    "A cat sleeps.": "Un chat dort.",
    "A dog does not run.": "Un chien ne court pas.",
}


def test_unique_sentences_cover_both_columns_once_longest_first() -> None:
    sentences = unique_sentences(_corpus())
    assert sorted(sentences) == sorted(TABLE)
    assert sentences[0] == "A dog does not run."


def test_translation_keeps_labels_and_metadata_on_their_pair() -> None:
    french = translated(_corpus(), TABLE)
    assert french["train"]["original"] == ["Un chien court.", "Un chat dort."]
    assert french["train"]["simplification"] == ["Un chien ne court pas.", "Un chien court."]
    assert french["train"]["polarity"] == [2, 0]
    assert french["train"]["corpus"] == ["sick", "lexical"]


def test_mixing_keeps_splits_apart() -> None:
    english = _corpus()
    both = mixed(english, translated(english, TABLE))
    assert set(both) == {"train", "test"}
    assert len(both["train"]) == 4
    assert both["test"]["original"][2] == "Un chien court."


def test_ready_marker_is_written_with_the_probes(tmp_path) -> None:
    source = tmp_path / "src"
    DatasetDict({"monli": Dataset.from_dict({"x": [1]})}).save_to_disk(str(source / "probes"))
    out = tmp_path / "fr"
    write(translated(_corpus(), TABLE), str(source), str(out))
    assert os.path.exists(out / "READY")
    assert os.path.exists(out / "probes" / "monli")
    assert os.path.exists(out / "corpus" / "dataset_dict.json")
