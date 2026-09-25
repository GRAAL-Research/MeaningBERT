"""Measure how much of its range a metric spends to separate agreement from contradiction.

The v3 acceptance criterion, from ROADMAP.md. LexFlip measured that on 373 perturbations
which reverse legal force while preserving 0.93 of the tokens, BERTScore and embedding
metrics move over 2 to 4 percent of their amplitude; only bidirectional NLI models move
(0.670). A metric that answers nearly the same number whether two sentences agree or
contradict is reporting lexical overlap, not meaning.

This runs on published checkpoints and trains nothing. It exists to put a number on the
failure before v3 tries to fix it, the same way v2 measured the sigmoid ceiling at 96.36
before replacing the head.

Two numbers per corpus:

**amplitude** is ``(mean on entailment) - (mean on contradiction)``, as a share of the
0-100 range. It is the headline: how much of its scale the metric actually spends on the
distinction.

**AUC** asks the weaker question: does the metric at least *rank* contradictions below
entailments, even by a hair. A metric can score 0.95 AUC while spending 3 points of
amplitude, which is precisely the failure LexFlip describes, so the two must be read
together.

Run::

    PYTHONPATH=src python src/diagnostics/dissociation.py \\
        --checkpoint davebulaval/MeaningBERT \\
        --checkpoint davebulaval/MeaningBERT --subfolder large
"""

from __future__ import annotations

import json
from typing import Optional

import click
import numpy as np
from scipy.stats import rankdata

try:  # PYTHONPATH=src.
    from meaningbert.scorer import MeaningBERTScorer
except ImportError:  # pragma: no cover
    from src.meaningbert.scorer import MeaningBERTScorer  # type: ignore


#: ``(dataset, config, split, left, right, label field, entailment values, contradiction values)``
#:
#: Only corpora that carry BOTH classes on sentence pairs are here. MoNLI has no
#: contradiction class, it opposes entailment to neutral, so its amplitude answers a
#: different and easier question; it is kept and labelled as such rather than silently
#: mixed into the same average.
SUITES = [
    # Verifie dans les donnees, pas devine : 0 est l'implication (les deux phrases sont
    # identiques dans le premier exemple), 2 la contradiction (« ... wrestling and hugging »
    # contre « There is no dog wrestling and hugging »). La correspondance inverse donnait
    # une AUC de 0,021, c'est-a-dire un classement presque parfaitement retourne, ce qui est
    # la signature d'une etiquette echangee et non d'un modele en echec.
    ("yangwang825/sick", None, "test", "text1", "text2", "label", {0}, {2}, "SICK"),
    ("joey234/nan-nli", None, "test", "premise", "hypothesis", "label",
     {"entailment"}, {"contradiction"}, "NaN-NLI (negation)"),
    ("tasksource/monli", None, "train", "sentence1", "sentence2", "gold_label",
     {"entailment"}, {"neutral"}, "MoNLI (entail vs neutral, NOT contradiction)"),
]


def amplitude(entail: np.ndarray, contra: np.ndarray) -> dict:
    """Separation between the two classes, plus the ranking check.

    The AUC uses **mid-ranks**, which is not a refinement here but a correctness
    requirement. A clamped output head saturates at exactly 0 and exactly 100, so ties
    between the two classes are not an edge case, they are the common case. Ordinal ranks
    (``argsort().argsort()``) break those ties by position in the array rather than
    declaring them undecided: two classes with identical scores on every pair came out at
    an AUC of 0.0 instead of 0.5, and simply swapping the two arguments moved the answer
    from 0.17 to 0.85. Mid-ranks give a tie its due half-credit, which is what the
    Mann-Whitney identity assumes.
    """
    n_e, n_c = len(entail), len(contra)
    # An empty class is a real outcome, not a crash: a filter can leave a suite with no
    # contradictions. numpy's mean of an empty slice is NaN but shouts a RuntimeWarning
    # while doing it, and a warning that fires on a handled case trains the reader to
    # ignore the ones that matter.
    mean_e = float(entail.mean()) if n_e else float("nan")
    mean_c = float(contra.mean()) if n_c else float("nan")
    gap = mean_e - mean_c

    # AUC by the rank identity: the share of (entailment, contradiction) couples the metric
    # orders correctly, counting a tie as half a couple. Computed from ranks so it costs
    # one sort instead of n*m comparisons.
    ranks = rankdata(np.concatenate([entail, contra]))
    auc = float((ranks[:n_e].sum() - n_e * (n_e + 1) / 2) / (n_e * n_c)) if n_e and n_c else float("nan")
    return {
        "n_entailment": n_e,
        "n_contradiction": n_c,
        "mean_entailment": mean_e,
        "mean_contradiction": mean_c,
        "amplitude_points": gap,
        "amplitude_share": gap / 100.0,
        "auc": auc,
    }


@click.command()
@click.option("--checkpoint", "checkpoints", multiple=True, required=True, help="Repeatable.")
@click.option("--subfolder", "subfolders", multiple=True, help="Aligned with --checkpoint; empty string for none.")
@click.option("--limit", default=600, show_default=True, help="Pairs per class, per corpus.")
@click.option("--json-out", default=None)
def main(checkpoints, subfolders, limit: int, json_out: Optional[str]) -> None:
    """Score every suite with every checkpoint and report the amplitude spent."""
    from datasets import load_dataset

    subfolders = list(subfolders) + [""] * (len(checkpoints) - len(subfolders))
    loaded = []
    for name, config, split, left, right, field, yes, no, label in SUITES:
        # Le decoupage annonce n'existe pas partout : nan-nli ne publie qu'un train. On
        # prend celui demande s'il existe, sinon le premier, plutot que d'echouer sur un
        # detail de publication.
        available = load_dataset(name, config) if config else load_dataset(name)
        chosen = split if split in available else list(available)[0]
        data = available[chosen]
        rows = {"entail": ([], []), "contra": ([], [])}
        for row in data:
            bucket = "entail" if row[field] in yes else "contra" if row[field] in no else None
            if bucket is None or len(rows[bucket][0]) >= limit:
                continue
            rows[bucket][0].append(row[left])
            rows[bucket][1].append(row[right])
        loaded.append((label, rows))
        print(f"{label:44} {len(rows['entail'][0]):4d} accord / {len(rows['contra'][0]):4d} contradiction"
              f"  ({chosen})")

    findings = []
    for checkpoint, subfolder in zip(checkpoints, subfolders):
        scorer = MeaningBERTScorer(checkpoint, subfolder=subfolder or None)
        tag = f"{checkpoint}{'/' + subfolder if subfolder else ''}"
        print(f"\n=== {tag}")
        for label, rows in loaded:
            entail = np.array(scorer.score(*rows["entail"]), dtype=float)
            contra = np.array(scorer.score(*rows["contra"]), dtype=float)
            got = amplitude(entail, contra)
            got.update({"checkpoint": tag, "suite": label})
            findings.append(got)
            print(f"  {label:44} accord {got['mean_entailment']:6.2f}  "
                  f"contradiction {got['mean_contradiction']:6.2f}  "
                  f"amplitude {got['amplitude_share'] * 100:5.1f} %  AUC {got['auc']:.3f}")
        del scorer

    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(findings, handle, indent=2)
        print(f"\nbrut : {json_out}")

    print("\nRepere LexFlip : les metriques a plongements consomment 2 a 4 % de leur amplitude,")
    print("les modeles NLI bidirectionnels 67 %. En dessous de 10 %, la metrique rapporte")
    print("du recouvrement lexical et non du sens.")


if __name__ == "__main__":
    main()
