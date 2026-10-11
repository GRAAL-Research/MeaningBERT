"""Translate the AUG+LEX polarity corpus to French, for the v4 translate-train regimes.

The v4 compares three ways of giving the polarity head French (``run_v4_queue.sh``): train
in English and read French zero-shot, train on the corpus translated to French, or train
on both. The last two need this corpus. Every column but the two sentences is kept, so a
translated row keeps its label, its source corpus and its split, and the leakage-free
split of the English corpus carries over unchanged.

The lexical pairs travel with the rest: an English antonym swap usually translates into a
French antonym swap. That is a first approximation of a French lexical augmentation, not a
native one (V4-H4), and the report says so.

Run::

    PYTHONPATH=src python src/data/translate_polarity_corpus.py \
        --source datastore/polarity-full-lex --out-fr datastore/polarity-full-lex-fr \
        --out-enfr datastore/polarity-full-lex-enfr
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Final

import click

TRANSLATOR: Final[str] = "Helsinki-NLP/opus-mt-en-fr"
SENTENCE_COLUMNS: Final[tuple[str, str]] = ("original", "simplification")
#: Flush the cache every so many sentences, so a killed run resumes instead of restarting.
FLUSH_EVERY: Final[int] = 20_000


def unique_sentences(splits) -> list[str]:
    """Every distinct sentence of the corpus, longest first so batches pad little."""
    seen: set[str] = set()
    for split in splits.values():
        for column in SENTENCE_COLUMNS:
            seen.update(split[column])
    return sorted(seen, key=len, reverse=True)


def translate_all(sentences: list[str], cache_path: str, batch_size: int, beams: int) -> dict[str, str]:
    """English to French, cached on disk and resumable."""
    cache: dict[str, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            cache = json.load(handle)
    todo = [s for s in sentences if s not in cache]
    click.echo(f"{len(sentences)} phrases, {len(cache)} deja traduites, {len(todo)} a faire")
    if not todo:
        return cache

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR).to(device).eval()

    def flush() -> None:
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False)
        os.replace(tmp, cache_path)

    done_since_flush = 0
    done = 0
    size = batch_size
    streak = 0
    while done < len(todo):
        batch = todo[done : done + size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        try:
            with torch.no_grad():
                generated = model.generate(**encoded, num_beams=beams, max_new_tokens=256)
        except torch.cuda.OutOfMemoryError:
            # Longest sentences come first, so the batch that fits grows as the run goes:
            # halve on failure, and double back after fifty batches in a row pass.
            torch.cuda.empty_cache()
            if size == 1:
                raise
            size = max(1, size // 2)
            streak = 0
            continue
        cache.update(zip(batch, tokenizer.batch_decode(generated, skip_special_tokens=True)))
        done += len(batch)
        done_since_flush += len(batch)
        streak += 1
        if streak >= 50:
            size, streak = min(batch_size, size * 2), 0
        if done_since_flush >= FLUSH_EVERY:
            flush()
            done_since_flush = 0
            click.echo(f"  {done} / {len(todo)}")
    flush()
    return cache


def translated(splits, table: dict[str, str]):
    """The same splits, sentences replaced by their translation."""
    return splits.map(lambda row: {column: table[row[column]] for column in SENTENCE_COLUMNS})


def mixed(english, french):
    """English and French rows of each split, concatenated."""
    from datasets import DatasetDict, concatenate_datasets

    return DatasetDict({name: concatenate_datasets([english[name], french[name]]) for name in english})


def write(splits, source: str, out: str) -> None:
    """Save *splits* as ``out/corpus``, copy the probes, and mark the corpus ready last."""
    if os.path.exists(out):
        shutil.rmtree(out)
    splits.save_to_disk(f"{out}/corpus")
    shutil.copytree(f"{source}/probes", f"{out}/probes")
    # READY is what run_v4_queue.sh waits for: written last, so a queue never trains on a
    # corpus that is still being saved.
    with open(f"{out}/READY", "w", encoding="utf-8") as handle:
        handle.write("ok\n")


@click.command()
@click.option("--source", default="datastore/polarity-full-lex", show_default=True)
@click.option("--out-fr", default="datastore/polarity-full-lex-fr", show_default=True)
@click.option("--out-enfr", default="datastore/polarity-full-lex-enfr", show_default=True)
@click.option("--cache", default="datastore/translations-en-fr.json", show_default=True)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--beams", default=4, show_default=True)
def main(source: str, out_fr: str, out_enfr: str, cache: str, batch_size: int, beams: int) -> None:
    """Traduit le corpus AUG+LEX en francais et ecrit les corpus fr et enfr."""
    from datasets import load_from_disk

    english = load_from_disk(f"{source}/corpus")
    table = translate_all(unique_sentences(english), cache, batch_size, beams)
    french = translated(english, table)
    write(french, source, out_fr)
    write(mixed(english, french), source, out_enfr)
    click.echo(f"corpus fr : {out_fr} ; corpus enfr : {out_enfr}")


if __name__ == "__main__":
    main()
