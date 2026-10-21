"""Tests for the multilingual XNLI report (``xnli_multi.py``).

The pairing across languages is the whole design: a head is compared with itself in
English on the same items. What would go wrong silently is two languages out of order, an
English head read natively, or a bar picked among trained heads.
"""

from __future__ import annotations

import numpy as np
import pytest

from diagnostics.xnli_multi import columns_of, is_native, misaligned, report, tables

LABELS = np.array(["entailment"] * 3 + ["neutral"] + ["contradiction"] * 3)
GOOD = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
NOISY = [0.4, 0.6, 0.3, 0.5, 0.5, 0.7, 0.45]


def test_a_few_swapped_labels_are_dropped_a_lost_order_is_refused() -> None:
    english = [{"label": i % 3} for i in range(300)]
    russian = [dict(row) for row in english]
    russian[10]["label"], russian[11]["label"] = russian[11]["label"], russian[10]["label"]
    assert misaligned(english, russian, "ru") == {10, 11}
    with pytest.raises(ValueError):
        misaligned(english, list(reversed(english)), "de")
    with pytest.raises(ValueError):
        misaligned(english, english[:-1], "de")


def test_only_multilingual_heads_read_the_language_directly() -> None:
    assert is_native("offshelf-xlm-roberta-large") and is_native("v4-mdeberta-en-seed42")
    assert not is_native("offshelf-deberta-v3-large") and not is_native("v3-auglex-seed42")


def test_seeds_average_into_an_ensemble_per_input() -> None:
    scores = {"v4-xlmr-en-seed42": {"de": [0.2, 0.4]}, "v4-xlmr-en-seed43": {"de": [0.4, 0.6]}}
    assert columns_of(scores)["v4-xlmr-en-ensemble/de"].tolist() == pytest.approx([0.3, 0.5])


def test_report_pairs_each_language_with_english_and_picks_untrained_bars() -> None:
    scores = {
        "offshelf-xlm-roberta-large": {"orig": GOOD, "de": NOISY},
        "offshelf-deberta-v3-large": {"orig": GOOD, "mt-de": GOOD},
        "v4-xlmr-en-seed42": {"orig": GOOD, "de": GOOD},
    }
    out = report(LABELS, scores, ("de",))
    assert out["auc"]["offshelf-xlm-roberta-large/orig"] == 1.0
    # The trained head beats the off-the-shelf one in German, and is still never the bar.
    assert out["bars"]["de"] == {"1a": "offshelf-xlm-roberta-large/de", "1b": "offshelf-deberta-v3-large/mt-de"}
    assert out["tests"]["v4-xlmr-en-ensemble/de"]["1a"]["diff"] > 0
    assert "p_holm" in out["tests"]["v4-xlmr-en-ensemble/de"]["1b"]
    # Same head, same items: a gap between English and German, zero for the faithful one.
    assert out["parity"]["offshelf-xlm-roberta-large/de"]["gap"] > 0
    assert out["parity"]["offshelf-deberta-v3-large/mt-de"]["gap"] == 0.0
    text = tables(out)
    assert "| `XLM-R-large NLI` | langue | 100.00 |" in text
    assert "| `DeBERTa-v3-large NLI` | traduit | 100.00 | 100.00 |" in text
    assert "réf." in text
