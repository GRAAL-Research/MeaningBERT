"""Translate a v2 magnitude variant to French, for the multilingual MeaningBERT benchmark.

The v4 magnitude benchmark (decision of 2026-10-21) trains the MeaningBERT recipe on
multilingual backbones and compares it, on FrJUDGE, with multilingual BERTScore. Like the
polarity grid it has a translate-train rung: the v2 variant (``d_full`` by default, the
corpus of the published v2-large) machine-translated to French, and English plus French.
Every column but the two sentences is kept, so a row keeps its score and its split; the
identical pairs stay identical, since one sentence always gets the same translation.

Run::

    PYTHONPATH=src python src/data/translate_magnitude_corpus.py \
        --source ~/MeaningBERT/data/v2/d_full --out-fr data/v4-magnitude/d_full-fr \
        --out-enfr data/v4-magnitude/d_full-enfr
"""

from __future__ import annotations

import os
import shutil

import click

try:
    from data.translate_polarity_corpus import mixed, translate_all, translated, unique_sentences
except ImportError:  # pragma: no cover - script run from inside ``src/data``.
    from translate_polarity_corpus import mixed, translate_all, translated, unique_sentences  # type: ignore


def save(splits, out: str) -> None:
    """Save *splits* at *out* and mark it ready last, as the training queues expect."""
    if os.path.exists(out):
        shutil.rmtree(out)
    splits.save_to_disk(out)
    with open(os.path.join(out, "READY"), "w", encoding="utf-8") as handle:
        handle.write("ok\n")


@click.command()
@click.option("--source", required=True, help="Dossier d'une variante v2 (DatasetDict).")
@click.option("--out-fr", required=True)
@click.option("--out-enfr", required=True)
@click.option("--cache", default="datastore/translations-en-fr.json", show_default=True)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--beams", default=4, show_default=True)
def main(source: str, out_fr: str, out_enfr: str, cache: str, batch_size: int, beams: int) -> None:
    """Traduit une variante v2 en francais et ecrit les variantes fr et enfr."""
    from datasets import load_from_disk

    english = load_from_disk(source)
    table = translate_all(unique_sentences(english), cache, batch_size, beams)
    french = translated(english, table)
    save(french, out_fr)
    save(mixed(english, french), out_enfr)
    click.echo(f"variante fr : {out_fr} ; variante enfr : {out_enfr}")


if __name__ == "__main__":
    main()
