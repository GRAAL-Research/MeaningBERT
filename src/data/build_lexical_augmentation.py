"""Lexical augmentation: antonym contradictions and synonym entailments for training.

The minimal pairs showed that the fine-tuned head catches an inserted ``not`` but almost
never an antonym swap (``outside`` / ``inside``), while the off-the-shelf head catches
most of them: training on VitaminC and SICK taught negation markers and lost opposition.
This adds the missing signal to the training split and nothing else.

* **Antonym contradictions**: a training sentence and the same sentence with one word
  replaced by its antonym, labelled contradiction.
* **Synonym entailments**: the same construction with a synonym or hypernym, labelled
  entailment, in equal number, so that the head cannot learn that any changed word is a
  contradiction.

The lexicons are split in two. Only the training half is used here; the minimal-pair
evaluation keeps the other half, so a gain on held-out antonyms cannot come from having
seen the very word pair. Sources are training sentences only, and a sentence that also
appears in the development or test split is never used. Run::

    PYTHONPATH=src python src/data/build_lexical_augmentation.py \\
        --corpus datastore/polarity/corpus --out datastore/polarity/corpus-lex
"""

from __future__ import annotations

import random
import re
from typing import Final

import click

try:
    from data.schema import POLARITY_CLASSES
    from diagnostics.minimal_pairs import ANTONYMS, HYPERNYMS, SYNONYMS
except ImportError:  # pragma: no cover - run from the repository root
    from src.data.schema import POLARITY_CLASSES  # type: ignore
    from src.diagnostics.minimal_pairs import ANTONYMS, HYPERNYMS, SYNONYMS  # type: ignore

#: Antonyms common in Wikipedia claims; disjoint from the caption lexicon of the test half.
FACTUAL_ANTONYMS: Final[tuple[tuple[str, str], ...]] = (
    ("increased", "decreased"),
    ("increase", "decrease"),
    ("more", "less"),
    ("higher", "lower"),
    ("won", "lost"),
    ("win", "lose"),
    ("first", "last"),
    ("larger", "smaller"),
    ("largest", "smallest"),
    ("earlier", "later"),
    ("positive", "negative"),
    ("rose", "fell"),
    ("maximum", "minimum"),
    ("above", "below"),
    ("north", "south"),
    ("east", "west"),
    ("male", "female"),
    ("public", "private"),
    ("major", "minor"),
    ("internal", "external"),
    ("success", "failure"),
    ("successful", "unsuccessful"),
    ("accepted", "rejected"),
    ("allowed", "banned"),
    ("born", "died"),
    ("best", "worst"),
    ("highest", "lowest"),
    ("most", "least"),
    ("older", "younger"),
    ("oldest", "youngest"),
)

#: Synonyms common in Wikipedia claims, for the entailment control.
FACTUAL_SYNONYMS: Final[tuple[tuple[str, str], ...]] = (
    ("film", "movie"),
    ("movie", "film"),
    ("began", "started"),
    ("started", "began"),
    ("purchased", "bought"),
    ("bought", "purchased"),
    ("approximately", "about"),
    ("about", "approximately"),
    ("received", "got"),
    ("big", "large"),
    ("large", "big"),
    ("famous", "well-known"),
    ("country", "nation"),
    ("nation", "country"),
    ("job", "position"),
    ("totally", "completely"),
    ("completely", "totally"),
    ("help", "assist"),
    ("people", "persons"),
    ("shown", "displayed"),
    ("aired", "broadcast"),
    ("broadcast", "aired"),
    ("died", "passed away"),
    ("wife", "spouse"),
    ("husband", "spouse"),
)


def split_half(pairs: tuple[tuple[str, str], ...]) -> tuple[tuple, tuple]:
    """Even positions train, odd positions test: deterministic and disjoint."""
    return pairs[0::2], pairs[1::2]


def antonym_table(pairs) -> dict[str, str]:
    table: dict[str, str] = {}
    for left, right in pairs:
        table[left] = right
        table.setdefault(right, left)
    return table


def train_tables() -> tuple[dict[str, str], dict[str, str]]:
    """The antonym and entailment tables the training half may use."""
    antonyms = antonym_table(split_half(ANTONYMS)[0] + FACTUAL_ANTONYMS)
    entailing = dict(split_half(SYNONYMS)[0] + split_half(HYPERNYMS)[0] + FACTUAL_SYNONYMS)
    return antonyms, entailing


def test_words() -> set[str]:
    """Every word of the held-out halves, for filtering the evaluation."""
    words = set()
    for pairs in (split_half(ANTONYMS)[1], split_half(SYNONYMS)[1], split_half(HYPERNYMS)[1]):
        for left, right in pairs:
            words |= {left, right}
    return words


def clean_edits(sentence: str, table: dict[str, str]) -> list[tuple[str, str, str]]:
    """Single-word edits, minus those inside a compound or a proper name.

    ``Change-Up`` is a title and ``Cheshire West`` a place: swapping the part changes a
    name, not a fact, so neither yields a contradiction or an entailment.
    """
    out = []
    for match in re.finditer(r"[A-Za-z]+", sentence):
        word, start, end = match.group(0), match.start(), match.end()
        new = table.get(word.lower())
        if new is None:
            continue
        if sentence[max(start - 1, 0) : start] == "-" or sentence[end : end + 1] == "-":
            continue
        if word[:1].isupper() and start > 0:
            continue
        replaced = new[:1].upper() + new[1:] if word[:1].isupper() else new
        out.append((sentence[:start] + replaced + sentence[end:], word, new))
    return out


def generate(sentences: list[str], table: dict[str, str], label: int, rng: random.Random, cap: int) -> list[dict]:
    """One edit per (sentence, word), labelled; at most ``cap`` rows."""
    rows = [
        {"original": sentence, "simplification": edited, "polarity": float(label), "old": old, "new": new}
        for sentence in sentences
        for edited, old, new in clean_edits(sentence, table)
    ]
    return rows if len(rows) <= cap else rng.sample(rows, cap)


@click.command()
@click.option("--corpus", default="datastore/polarity/corpus", show_default=True)
@click.option("--out", required=True)
@click.option("--per-kind", default=20_000, show_default=True)
@click.option("--seed", default=42, show_default=True)
def main(corpus: str, out: str, per_kind: int, seed: int) -> None:
    """Write a copy of the corpus whose training split carries the lexical pairs."""
    from datasets import concatenate_datasets, load_from_disk

    rng = random.Random(seed)
    splits = load_from_disk(corpus)
    held = set()
    for name in ("dev", "test"):
        held |= set(splits[name]["original"]) | set(splits[name]["simplification"])
    train = splits["train"]
    # The claim side of VitaminC and both sides of SICK: short, single-fact sentences.
    sentences = sorted(
        {
            text
            for row in train
            for text in (
                (row["simplification"],) if row["corpus"] == "vitaminc" else (row["original"], row["simplification"])
            )
            if text not in held and len(text.split()) <= 40
        }
    )
    antonyms, entailing = train_tables()
    contra = generate(sentences, antonyms, POLARITY_CLASSES["contradiction"], rng, per_kind)
    entail = generate(sentences, entailing, POLARITY_CLASSES["entailment"], rng, len(contra))
    template = train[0]
    added = []
    for index, row in enumerate(contra + entail):
        new = {key: template[key] for key in train.column_names}
        new.update(
            {
                "item_id": f"lexical:{index}",
                "original": row["original"],
                "simplification": row["simplification"],
                "polarity": row["polarity"],
                "corpus": "lexical",
                "source": "generated",
                "system": "lexical",
                "polarity_raw": (
                    "contradiction" if row["polarity"] == POLARITY_CLASSES["contradiction"] else "entailment"
                ),
            }
        )
        added.append(new)
    from datasets import Dataset

    extra = Dataset.from_list(added, features=train.features)
    splits["train"] = concatenate_datasets([train, extra]).shuffle(seed=seed)
    splits.save_to_disk(out)
    click.echo(
        f"{len(contra)} contradictions par antonyme, {len(entail)} implications ; train = {len(splits['train'])}"
    )


if __name__ == "__main__":
    main()
