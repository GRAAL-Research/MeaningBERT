"""Generate a loader's delivery report from the data it actually produces.

``CONTRACT.md`` requires one ``<corpus>.report.json`` per loader. The four v2 reports were
written by hand, which was affordable at four and is not at nine: a hand-written census
rots in silence, and the first thing that rots is the number nobody re-checked. Same
reasoning as ``src/diagnostics/releve_corpus_v3.py``, which is why this exists.

It also widens the report to the second target. A v3 corpus is described by what it does
NOT annotate as much as by what it does, so the report carries ``polarity_scheme`` and the
class census beside the label statistics.

Run::

    PYTHONPATH=src python src/data/loaders/report.py --corpus sick --corpus monli
"""

from __future__ import annotations

import collections
import importlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Optional

import click
from datasets import Dataset

from data.schema import validate

#: Where each corpus was retrieved from, for the ``source_url`` field. A report that does
#: not say where the bytes came from cannot be re-checked.
SOURCE_URLS: dict[str, str] = {
    "sick": "https://huggingface.co/datasets/yangwang825/sick + https://huggingface.co/datasets/mteb/sickr-sts",
    "vitaminc": "https://huggingface.co/datasets/tals/vitaminc",
    "monli": "https://huggingface.co/datasets/tasksource/monli",
    "nan_nli": "https://huggingface.co/datasets/joey234/nan-nli",
}

#: Answers H4 of PRODUIT.md per corpus: may the merged corpus ship the rows, or only the
#: loader? ``False`` wherever the upstream licence is unstated or unread, which is the
#: conservative reading and costs nothing here: the v3 merge goes out under the strictest
#: licence of its inputs regardless.
REDISTRIBUTION: dict[str, bool] = {
    "sick": False,
    "vitaminc": True,
    "monli": False,
    "nan_nli": True,
}


def _finite(values: list[float]) -> list[float]:
    """Keep only the values that are actual measurements."""
    return [value for value in values if isinstance(value, float) and math.isfinite(value)]


def build_report(dataset: Dataset, corpus: str, notes: str = "") -> dict:
    """Measure *dataset* into the report shape ``CONTRACT.md`` describes.

    Args:
        dataset: A validated loader output.
        corpus: The corpus identifier, also the report's file stem.
        notes: Mapping decisions, dropped rows and why. Written by the author, not
            measured, because only the author knows what was decided.

    Returns:
        The report as a plain dict, ready to serialise.
    """
    labels = _finite(list(dataset["label_raw"]))
    annotators = sorted(dataset["n_annotators"])
    with_std = sum(1 for value in dataset["label_std"] if isinstance(value, float) and math.isfinite(value))
    scheme = sorted(set(dataset["polarity_scheme"]))

    return {
        "corpus": corpus,
        "n_rows": len(dataset),
        "n_unique_originals": len(set(dataset["original"])),
        "scale": sorted(set(dataset["scale"]))[0],
        # A polarity-only corpus has no label statistics, and reporting 0.0 would read as a
        # measurement. None is the honest answer to a question that was never asked.
        "label_raw_min": min(labels) if labels else None,
        "label_raw_max": max(labels) if labels else None,
        "label_raw_mean": sum(labels) / len(labels) if labels else None,
        "n_annotators_median": annotators[len(annotators) // 2] if annotators else 0,
        "pct_with_std": round(100.0 * with_std / len(dataset), 2),
        "polarity_scheme": scheme[0] if len(scheme) == 1 else scheme,
        "polarity_classes": dict(collections.Counter(dataset["polarity_raw"]).most_common()),
        "domains": dict(collections.Counter(dataset["domain"]).most_common()),
        "systems": dict(collections.Counter(dataset["system"]).most_common(20)),
        "split_hints": dict(collections.Counter(dataset["split_hint"]).most_common()),
        "license": sorted(set(dataset["license"]))[0],
        "license_allows_redistribution": REDISTRIBUTION.get(corpus, False),
        "source_url": SOURCE_URLS.get(corpus, ""),
        "retrieval_date": date.today().isoformat(),
        "notes": notes,
    }


@click.command()
@click.option("--corpus", "corpora", multiple=True, required=True, help="Repeatable loader module name.")
@click.option("--notes-from", default=None, help="Directory of <corpus>.notes.txt files to inline.")
def main(corpora: tuple[str, ...], notes_from: Optional[str]) -> None:
    """Run each loader, validate it, and write its report next to it."""
    here = Path(__file__).resolve().parent
    for corpus in corpora:
        click.echo(f"--- {corpus}")
        dataset = importlib.import_module(f"data.loaders.{corpus}").load()
        validate(dataset)

        notes = ""
        if notes_from:
            candidate = Path(notes_from) / f"{corpus}.notes.txt"
            if candidate.exists():
                notes = candidate.read_text(encoding="utf-8").strip()

        report = build_report(dataset, corpus, notes)
        destination = here / f"{corpus}.report.json"
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        click.echo(
            f"    {report['n_rows']:>7,} lignes | echelle {report['scale']} | "
            f"polarite {report['polarity_scheme']} | {report['polarity_classes']}"
        )
        click.echo(f"    rapport : {destination}")


if __name__ == "__main__":
    main()
