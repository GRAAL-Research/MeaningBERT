"""Minimal pairs: change one word at a time and watch the signed score move.

The composition is credited with placing contradictions below zero, but on SICK most
contradictions carry an explicit negation, which a word list would catch as well. This
asks the question directly. Every pair differs from a SICK test sentence by a single
word, and the kind of edit fixes what the score should do:

* ``negation``: ``is`` becomes ``is not``. A contradiction, and the only kind a list of
  negation words flags.
* ``antonym``: ``outside`` becomes ``inside``. A contradiction under SICK's convention
  that both sentences describe the same scene, with no negation word anywhere.
* ``synonym`` and ``hypernym``: ``sofa`` becomes ``couch``, ``dog`` becomes ``animal``.
  The meaning survives, so the score should stay positive.
* ``cohyponym``: ``dog`` becomes ``cat``. Another entity in the same scene; the
  convention is less settled, so it is reported, not scored against an expected sign.

A ``chain`` then applies antonym and co-hyponym edits one after the other to the same
sentence, so the score can be read after one, two, three and four changed words.

The lexicons are written by hand for SICK's vocabulary, which is small and concrete:
captions of people, animals and everyday actions. Run::

    PYTHONPATH=src python src/diagnostics/minimal_pairs.py build --out results/v3/minimal-pairs.json
    PYTHONPATH=src python src/diagnostics/minimal_pairs.py score --pairs results/v3/minimal-pairs.json \\
        --polarity results/polarity/nli-deberta-v3-large-none-poids/seed42/model \\
        --out results/v3/minimal-pairs-seed42.json
"""

from __future__ import annotations

import json
import random
import re
from typing import Final, Iterable, Optional

import click
import numpy as np

#: Pairs read both ways; a contradiction under the same-scene convention.
ANTONYMS: Final[tuple[tuple[str, str], ...]] = (
    ("inside", "outside"),
    ("indoors", "outdoors"),
    ("up", "down"),
    ("wet", "dry"),
    ("standing", "sitting"),
    ("empty", "full"),
    ("young", "old"),
    ("happy", "sad"),
    ("smiling", "frowning"),
    ("laughing", "crying"),
    ("slowly", "quickly"),
    ("day", "night"),
    ("hot", "cold"),
    ("opening", "closing"),
    ("pushing", "pulling"),
    ("buying", "selling"),
    ("starting", "stopping"),
    ("winning", "losing"),
    ("asleep", "awake"),
    ("entering", "leaving"),
    ("raising", "lowering"),
    ("adding", "removing"),
    ("clean", "dirty"),
    ("loudly", "quietly"),
    ("tall", "short"),
    ("thin", "thick"),
    ("dark", "bright"),
    ("light", "heavy"),
    ("top", "bottom"),
    ("left", "right"),
    ("near", "far"),
    ("before", "after"),
    ("ascending", "descending"),
    ("alone", "together"),
    ("many", "few"),
    ("large", "small"),
    ("big", "little"),
)

#: One direction only; the meaning survives the swap.
SYNONYMS: Final[tuple[tuple[str, str], ...]] = (
    ("sofa", "couch"),
    ("couch", "sofa"),
    ("big", "large"),
    ("large", "big"),
    ("small", "little"),
    ("little", "small"),
    ("quickly", "rapidly"),
    ("street", "road"),
    ("road", "street"),
    ("talking", "speaking"),
    ("speaking", "talking"),
    ("shouting", "yelling"),
    ("yelling", "shouting"),
    ("jumping", "leaping"),
    ("cutting", "slicing"),
    ("slicing", "cutting"),
    ("picture", "photo"),
    ("photo", "picture"),
    ("stone", "rock"),
    ("rock", "stone"),
    ("motorbike", "motorcycle"),
    ("motorcycle", "motorbike"),
    ("bike", "bicycle"),
    ("bicycle", "bike"),
    ("kid", "child"),
    ("child", "kid"),
    ("kids", "children"),
    ("children", "kids"),
    ("automobile", "car"),
    ("beach", "shore"),
    ("woman", "lady"),
    ("lady", "woman"),
    ("man", "guy"),
    ("guy", "man"),
    ("puppy", "pup"),
)

#: Specific to general; the general sentence follows from the specific one.
HYPERNYMS: Final[tuple[tuple[str, str], ...]] = (
    ("dog", "animal"),
    ("cat", "animal"),
    ("horse", "animal"),
    ("puppy", "dog"),
    ("kitten", "cat"),
    ("man", "person"),
    ("woman", "person"),
    ("boy", "child"),
    ("girl", "child"),
    ("guitar", "instrument"),
    ("piano", "instrument"),
    ("violin", "instrument"),
    ("flute", "instrument"),
    ("drums", "instruments"),
    ("car", "vehicle"),
    ("truck", "vehicle"),
    ("motorcycle", "vehicle"),
    ("bicycle", "vehicle"),
    ("sandwich", "food"),
    ("apple", "fruit"),
    ("banana", "fruit"),
    ("onion", "vegetable"),
    ("potato", "vegetable"),
    ("rose", "flower"),
    ("sprinting", "running"),
    ("jogging", "running"),
    ("sprints", "runs"),
)

#: Another member of the same category: a different entity in the same scene.
COHYPONYMS: Final[tuple[tuple[str, str], ...]] = (
    ("dog", "cat"),
    ("cat", "dog"),
    ("horse", "cow"),
    ("man", "woman"),
    ("woman", "man"),
    ("boy", "girl"),
    ("girl", "boy"),
    ("men", "women"),
    ("women", "men"),
    ("boys", "girls"),
    ("girls", "boys"),
    ("guitar", "piano"),
    ("piano", "guitar"),
    ("violin", "flute"),
    ("car", "truck"),
    ("truck", "car"),
    ("apple", "orange"),
    ("onion", "potato"),
    ("potato", "onion"),
    ("red", "blue"),
    ("blue", "green"),
    ("green", "red"),
    ("black", "white"),
    ("white", "black"),
    ("yellow", "red"),
    ("running", "walking"),
    ("walking", "running"),
    ("swimming", "running"),
    ("dancing", "singing"),
    ("singing", "dancing"),
    ("playing", "watching"),
    ("eating", "cooking"),
    ("cooking", "eating"),
    ("riding", "pushing"),
    ("beach", "park"),
    ("park", "beach"),
    ("snow", "sand"),
    ("water", "grass"),
    ("grass", "water"),
)

#: The negation the edit inserts after the first copula.
COPULAS: Final[tuple[str, ...]] = ("is", "are")

#: What each kind should do to the sign; None when the convention is not settled.
EXPECTED: Final[dict[str, Optional[str]]] = {
    "negation": "contradiction",
    "antonym": "contradiction",
    "synonym": "entailment",
    "hypernym": "entailment",
    "cohyponym": None,
}


def lexicon(kind: str) -> dict[str, str]:
    """The substitution table for one kind; antonyms are read in both directions."""
    if kind == "antonym":
        table = {}
        for left, right in ANTONYMS:
            table[left] = right
            table.setdefault(right, left)
        return table
    pairs = {"synonym": SYNONYMS, "hypernym": HYPERNYMS, "cohyponym": COHYPONYMS}[kind]
    return dict(pairs)


def _match_case(original: str, replacement: str) -> str:
    return replacement[:1].upper() + replacement[1:] if original[:1].isupper() else replacement


def substitute(sentence: str, table: dict[str, str]) -> list[tuple[str, str, str]]:
    """Every single-word substitution the table allows, as (edited, old, new).

    Whole words only, so ``man`` does not fire inside ``woman``, and one occurrence at
    a time, so each edited sentence differs from its source by exactly one word.
    """
    out = []
    for match in re.finditer(r"[A-Za-z]+", sentence):
        word = match.group(0)
        new = table.get(word.lower())
        if new is None:
            continue
        edited = sentence[: match.start()] + _match_case(word, new) + sentence[match.end() :]
        out.append((edited, word, new))
    return out


def negate(sentence: str) -> Optional[str]:
    """Insert ``not`` after the first copula; None when there is none to negate.

    A sentence that is already negated is left out, since a second negation would
    turn a contradiction back into an agreement.
    """
    if re.search(r"\b(not|no|never|nobody|none|nothing)\b|n't", sentence, re.IGNORECASE):
        return None
    match = re.search(r"\b(" + "|".join(COPULAS) + r")\b", sentence)
    if match is None:
        return None
    return sentence[: match.end()] + " not" + sentence[match.end() :]


def chain(sentence: str, rng: random.Random, steps: int = 4) -> Optional[list[str]]:
    """Successive antonym and co-hyponym edits to one sentence, one word per step.

    Returns the sentence after each step, or None when fewer than ``steps`` distinct
    words can be changed. Each step edits a word no earlier step touched, so step k
    differs from the source by exactly k words.
    """
    tables = {**lexicon("cohyponym"), **lexicon("antonym")}
    spans = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[A-Za-z]+", sentence)]
    editable = [span for span in spans if span[2].lower() in tables]
    if len(editable) < steps:
        return None
    order = rng.sample(editable, steps)
    current, versions, shift = sentence, [], {}
    for start, end, word in order:
        offset = sum(delta for position, delta in shift.items() if position < start)
        new = _match_case(word, tables[word.lower()])
        current = current[: start + offset] + new + current[end + offset :]
        shift[start] = len(new) - len(word)
        versions.append(current)
    return versions


def build_pairs(sentences: Iterable[str], seed: int = 42, per_kind: int = 2000) -> list[dict]:
    """All minimal pairs, at most ``per_kind`` per kind, and the edit chains."""
    rng = random.Random(seed)
    unique = sorted(set(sentences))
    by_kind: dict[str, list[dict]] = {kind: [] for kind in EXPECTED}
    for sentence in unique:
        negated = negate(sentence)
        if negated:
            by_kind["negation"].append({"kind": "negation", "source": sentence, "edited": negated, "step": 1})
        for kind in ("antonym", "synonym", "hypernym", "cohyponym"):
            for edited, old, new in substitute(sentence, lexicon(kind)):
                by_kind[kind].append(
                    {"kind": kind, "source": sentence, "edited": edited, "step": 1, "old": old, "new": new}
                )
    pairs = []
    for kind, rows in by_kind.items():
        pairs += rows if len(rows) <= per_kind else rng.sample(rows, per_kind)
    for index, sentence in enumerate(unique):
        versions = chain(sentence, rng)
        if versions:
            pairs += [
                {"kind": "chain", "source": sentence, "edited": edited, "step": step, "chain": index}
                for step, edited in enumerate(versions, start=1)
            ]
    return pairs


@click.group()
def cli() -> None:
    """Build, then score, the minimal pairs."""


@cli.command()
@click.option("--corpus", default="datastore/polarity/corpus", show_default=True)
@click.option("--split", default="test", show_default=True)
@click.option("--out", required=True)
@click.option("--per-kind", default=2000, show_default=True)
def build(corpus: str, split: str, out: str, per_kind: int) -> None:
    """Draw the source sentences from SICK and write every minimal pair."""
    from datasets import load_from_disk

    rows = load_from_disk(corpus)[split].filter(lambda row: row["corpus"] == "sick")
    pairs = build_pairs(list(rows["original"]) + list(rows["simplification"]), per_kind=per_kind)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(pairs, handle)
    counts = {}
    for pair in pairs:
        counts[pair["kind"]] = counts.get(pair["kind"], 0) + 1
    click.echo(f"{len(pairs)} paires : {counts}")


@cli.command()
@click.option("--pairs", "pairs_path", required=True)
@click.option("--magnitude", default="davebulaval/MeaningBERT", show_default=True)
@click.option("--magnitude-subfolder", default="large", show_default=True)
@click.option("--polarity", required=True)
@click.option("--out", required=True)
def score(pairs_path: str, magnitude: str, magnitude_subfolder: str, polarity: str, out: str) -> None:
    """Similarity, contradiction probability and signed score for every pair."""
    from diagnostics.composition import _probabilities, compose
    from diagnostics.cross_task import Model

    try:
        from data.schema import POLARITY_CLASSES
    except ImportError:  # pragma: no cover - run from the repository root
        from src.data.schema import POLARITY_CLASSES  # type: ignore

    with open(pairs_path, encoding="utf-8") as handle:
        pairs = json.load(handle)
    left = [pair["source"] for pair in pairs]
    right = [pair["edited"] for pair in pairs]

    similarity = Model(magnitude, magnitude_subfolder or None)
    m = similarity.meaning_score(similarity.logits(left, right))
    del similarity
    head = Model(polarity, None)
    p_contra = _probabilities(head.logits(left, right))[:, POLARITY_CLASSES["contradiction"]]
    signed = compose(m, p_contra, 2.0)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "polarity_checkpoint": polarity,
                "magnitude_checkpoint": f"{magnitude}/{magnitude_subfolder}",
                "magnitude": [float(v) for v in m],
                "p_contradiction": [float(v) for v in p_contra],
                "signed": [float(v) for v in np.asarray(signed)],
            },
            handle,
        )
    click.echo(f"{len(pairs)} paires scorees : {out}")


if __name__ == "__main__":
    cli()
