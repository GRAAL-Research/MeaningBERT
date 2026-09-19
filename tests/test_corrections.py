"""Tests for the documented corpus corrections."""

import numpy as np
import pytest

from data.corrections import CORRECTIONS, apply_corrections
from data.harmonize import find_anchors
from data.schema import build


def _corpus(name, rows, item_prefix=""):
    return build(
        [
            {
                "item_id": f"{item_prefix}{index}",
                "original": original,
                "simplification": simplification,
                "label_raw": float(value),
                "scale": "da100",
                "domain": "wiki",
                "license": "MIT",
            }
            for index, (original, simplification, value) in enumerate(rows)
        ],
        name,
    )


def _pair(i):
    return (f"source {i}", f"simplification {i}")


def _scrambled_case(n=8):
    """CSMD holding the right multiset of labels attached to the wrong pairs."""
    truth = [float(10 * i) for i in range(n)]
    scrambled = truth[::-1]
    csmd = _corpus("csmd", [(*_pair(i), scrambled[i]) for i in range(n)])
    simpeval = _corpus("simpeval", [(*_pair(i), truth[i]) for i in range(n)], item_prefix="2022-")
    return {"csmd": csmd, "simpeval": simpeval}, truth


def test_the_correction_restores_agreement_with_the_authoritative_source():
    corpora, truth = _scrambled_case()
    before = find_anchors(corpora["simpeval"], corpora["csmd"]).pearson
    fixed, _ = apply_corrections(corpora)
    after = find_anchors(fixed["simpeval"], fixed["csmd"]).pearson
    assert before < 0.0
    assert after == pytest.approx(1.0)


def test_the_correction_reports_how_many_rows_it_touched():
    corpora, _ = _scrambled_case(n=8)
    _, applied = apply_corrections(corpora)
    assert applied["csmd-simpda2022-relabel"] == 8


def test_the_correction_leaves_the_label_multiset_unchanged():
    """A re-pairing moves labels between rows; it must not invent or drop any value."""
    corpora, _ = _scrambled_case()
    fixed, _ = apply_corrections(corpora)
    assert sorted(fixed["csmd"]["label_raw"]) == sorted(corpora["csmd"]["label_raw"])


def test_the_correction_leaves_the_corpus_mean_and_spread_unchanged():
    corpora, _ = _scrambled_case()
    fixed, _ = apply_corrections(corpora)
    before = np.array(corpora["csmd"]["label_raw"])
    after = np.array(fixed["csmd"]["label_raw"])
    assert after.mean() == pytest.approx(before.mean())
    assert after.std() == pytest.approx(before.std())


def test_only_the_2022_subset_is_authoritative():
    """simpeval_past agrees with CSMD at r=0.68 and must not override it."""
    csmd = _corpus("csmd", [(*_pair(0), 10.0)])
    simpeval = _corpus("simpeval", [(*_pair(0), 90.0)], item_prefix="past-")
    fixed, applied = apply_corrections({"csmd": csmd, "simpeval": simpeval})
    assert applied["csmd-simpda2022-relabel"] == 0
    assert fixed["csmd"]["label_raw"] == [10.0]


def test_pairs_absent_from_the_authority_keep_their_csmd_label():
    csmd = _corpus("csmd", [(*_pair(0), 10.0), ("lonely", "LONELY", 42.0)])
    simpeval = _corpus("simpeval", [(*_pair(0), 90.0)], item_prefix="2022-")
    fixed, _ = apply_corrections({"csmd": csmd, "simpeval": simpeval})
    assert fixed["csmd"]["label_raw"] == [90.0, 42.0]


def test_no_text_is_altered():
    corpora, _ = _scrambled_case()
    fixed, _ = apply_corrections(corpora)
    assert fixed["csmd"]["original"] == corpora["csmd"]["original"]
    assert fixed["csmd"]["simplification"] == corpora["csmd"]["simplification"]


def test_no_row_is_added_or_dropped():
    corpora, _ = _scrambled_case()
    fixed, _ = apply_corrections(corpora)
    assert len(fixed["csmd"]) == len(corpora["csmd"])


def test_the_correction_is_a_no_op_without_the_authoritative_corpus():
    csmd = _corpus("csmd", [(*_pair(0), 10.0)])
    fixed, applied = apply_corrections({"csmd": csmd})
    assert applied["csmd-simpda2022-relabel"] == 0
    assert fixed["csmd"]["label_raw"] == [10.0]


def test_the_correction_is_idempotent():
    corpora, _ = _scrambled_case()
    once, _ = apply_corrections(corpora)
    twice, applied = apply_corrections(once)
    assert once["csmd"]["label_raw"] == twice["csmd"]["label_raw"]
    assert applied["csmd-simpda2022-relabel"] == len(once["csmd"])


def test_every_correction_carries_reproducible_evidence():
    """A correction without evidence is a preference, and does not belong here."""
    for correction in CORRECTIONS:
        assert len(correction.evidence) > 120
        assert "docs/" in correction.evidence
