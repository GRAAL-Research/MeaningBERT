"""Tests for the loader delivery-report generator (``src/data/loaders/report.py``).

The reports are what ``CONTRACT.md`` requires a loader to ship with, and they are the only
written record of what each corpus actually contains. The v2 reports were written by hand;
these are measured, which is the point, so the measurement itself has to be right.
"""

from __future__ import annotations

from data.loaders.report import REDISTRIBUTION, SOURCE_URLS, build_report
from data.schema import build


def _dataset(rows, corpus="demo"):
    return build(rows, corpus)


def _magnitude_row(index, label, **overrides):
    row = {
        "item_id": str(index),
        "original": f"source sentence {index}",
        "simplification": f"candidate {index}",
        "label_raw": label,
        "scale": "da100",
        "domain": "wiki",
        "license": "MIT",
        "n_annotators": 3,
        "label_std": 4.0,
    }
    row.update(overrides)
    return row


def _polarity_row(index, polarity, **overrides):
    row = {
        "item_id": str(index),
        "original": f"premise {index}",
        "simplification": f"hypothesis {index}",
        "label_raw": float("nan"),
        "scale": "none",
        "domain": "wiki",
        "license": "CC-BY-SA-3.0",
        "polarity_raw": polarity,
        "polarity_scheme": "nli3",
    }
    row.update(overrides)
    return row


# --- the label statistics ------------------------------------------------------------


def test_the_label_statistics_describe_the_annotated_values():
    dataset = _dataset([_magnitude_row(0, 20.0), _magnitude_row(1, 80.0), _magnitude_row(2, 50.0)])
    report = build_report(dataset, "demo")
    assert report["label_raw_min"] == 20.0
    assert report["label_raw_max"] == 80.0
    assert report["label_raw_mean"] == 50.0
    assert report["n_rows"] == 3


def test_a_polarity_only_corpus_reports_no_label_statistics_rather_than_zero():
    # Reporting 0.0 would read as a measurement: "the mean preservation score is zero".
    # The honest answer to a question nobody asked is that there is no answer.
    dataset = _dataset([_polarity_row(0, "entailment"), _polarity_row(1, "contradiction")])
    report = build_report(dataset, "vitaminc")
    assert report["label_raw_min"] is None
    assert report["label_raw_max"] is None
    assert report["label_raw_mean"] is None
    assert report["scale"] == "none"


def test_a_nan_label_is_left_out_of_the_statistics_it_would_poison():
    dataset = _dataset([_magnitude_row(0, 20.0), _magnitude_row(1, 80.0)])
    report = build_report(dataset, "demo")
    assert report["label_raw_mean"] == 50.0


# --- the polarity census -------------------------------------------------------------


def test_the_polarity_classes_are_counted_by_name():
    dataset = _dataset(
        [_polarity_row(0, "entailment"), _polarity_row(1, "contradiction"), _polarity_row(2, "contradiction")]
    )
    report = build_report(dataset, "vitaminc")
    assert report["polarity_classes"] == {"contradiction": 2, "entailment": 1}
    assert report["polarity_scheme"] == "nli3"


def test_a_corpus_without_polarity_says_so_explicitly():
    report = build_report(_dataset([_magnitude_row(0, 60.0)]), "demo")
    assert report["polarity_scheme"] == "none"
    assert report["polarity_classes"] == {"": 1}


# --- provenance, which is what makes a report re-checkable ---------------------------


def test_each_declared_corpus_carries_a_source_url_and_a_redistribution_verdict():
    # A report that does not say where the bytes came from cannot be re-checked, and one
    # that does not answer H4 cannot tell the merge what it may ship.
    for corpus in ("sick", "vitaminc", "paws", "monli", "nan_nli"):
        assert SOURCE_URLS[corpus].startswith("https://")
        assert corpus in REDISTRIBUTION


def test_an_unlisted_corpus_defaults_to_refusing_redistribution():
    # The conservative direction: an unknown licence must never read as permission.
    report = build_report(_dataset([_magnitude_row(0, 60.0)]), "corpus_inconnu")
    assert report["license_allows_redistribution"] is False


def test_the_report_carries_the_corpus_licence_and_row_census():
    dataset = _dataset([_polarity_row(0, "entailment"), _polarity_row(1, "entailment", domain="news")])
    report = build_report(dataset, "vitaminc")
    assert report["license"] == "CC-BY-SA-3.0"
    assert report["domains"] == {"wiki": 1, "news": 1}
    assert report["n_unique_originals"] == 2


def test_the_author_notes_are_carried_through_verbatim():
    # The notes are the one field that cannot be measured: only the author knows what was
    # decided and what was dropped.
    report = build_report(_dataset([_magnitude_row(0, 60.0)]), "demo", notes="deux lignes ecartees")
    assert report["notes"] == "deux lignes ecartees"
