"""Baselines that bound what the polarity head's 91 points actually mean.

A three-way score on a balanced split is only informative against something. Three
references, cheap enough that there is no excuse for omitting them:

**Majority class.** The floor. On a split balanced at 4,000 per class it is 33.3 points of
accuracy and 16.7 of macro-F1, and any number near it says the head learned nothing.

**Lexical overlap.** The question the paper exists to ask, asked of a baseline: can a
monotone function of token overlap separate entailment from contradiction? A
contradiction shares most of its tokens with what it contradicts, so a threshold on
overlap should sit near chance on the entailment-contradiction pair while doing well on
the neutral one. Measuring that is what turns a claim about magnitude metrics into a
measurement.

**TF-IDF plus logistic regression.** A bag of word pairs with no notion of order,
negation or scope. Whatever it reaches is the part of the task that is surface statistics
of the corpus, and the gap to the fine-tuned head is the part that is not.

All three run on CPU in under two minutes::

    PYTHONPATH=src python src/diagnostics/baselines_polarity.py \
        --corpus datastore/polarity --json-out results/v3/baselines.json
"""

from __future__ import annotations

import json
import re
from typing import Optional

import click
import numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

#: Unified polarity codes, as ``src/data/schema.py`` writes them.
ENTAILMENT, NEUTRAL, CONTRADICTION = 0, 1, 2

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def overlap(original: str, simplification: str) -> float:
    """Jaccard overlap of the two token sets.

    Jaccard and not containment: containment is asymmetric, and a baseline meant to stand
    in for a magnitude metric should not encode a direction the metric does not have.
    """
    left, right = tokens(original), tokens(simplification)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def scores(split) -> np.ndarray:
    return np.array([overlap(o, s) for o, s in zip(split["original"], split["simplification"])])


def labels(split) -> np.ndarray:
    return np.array([int(p) for p in split["polarity"]])


def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    return float(f1_score(truth, predicted, average="macro"))


def majority_baseline(train, test) -> dict:
    """Predict the most frequent training class, everywhere."""
    counts = np.bincount(labels(train), minlength=3)
    call = int(np.argmax(counts))
    predicted = np.full(len(test), call)
    truth = labels(test)
    return {
        "predicted_class": call,
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": macro_f1(truth, predicted),
    }


def overlap_baseline(train, dev, test) -> dict:
    """Two thresholds on token overlap, fitted on development data.

    Ordering the classes by expected overlap -- neutral lowest, then the two that share a
    subject -- is the only structure this baseline has. The thresholds are chosen by a
    grid search on dev macro-F1 so the baseline is given its best chance rather than a
    convenient one, and the AUC is reported separately because ranking is what the
    composition consumes.
    """
    dev_scores, dev_truth = scores(dev), labels(dev)
    grid = np.quantile(scores(train), np.linspace(0.02, 0.98, 49))
    best = (-1.0, grid[0], grid[-1])
    for low in grid:
        for high in grid:
            if high <= low:
                continue
            predicted = np.where(dev_scores < low, NEUTRAL, np.where(dev_scores < high, CONTRADICTION, ENTAILMENT))
            value = macro_f1(dev_truth, predicted)
            if value > best[0]:
                best = (value, float(low), float(high))
    _, low, high = best
    test_scores, test_truth = scores(test), labels(test)
    predicted = np.where(test_scores < low, NEUTRAL, np.where(test_scores < high, CONTRADICTION, ENTAILMENT))
    return {
        "low": low,
        "high": high,
        "dev_macro_f1": best[0],
        "accuracy": float(accuracy_score(test_truth, predicted)),
        "macro_f1": macro_f1(test_truth, predicted),
        "auc_entailment_vs_contradiction": auc_entail_vs_contra(test_scores, test_truth),
    }


def auc_entail_vs_contra(score: np.ndarray, truth: np.ndarray) -> float:
    """How well a score ranks entailment above contradiction, neutral pairs excluded.

    This is the quantity the signed scale consumes, and the one a magnitude metric is
    expected to be near chance on: both classes share a subject and most of their tokens.
    """
    keep = (truth == ENTAILMENT) | (truth == CONTRADICTION)
    if keep.sum() == 0 or len(set(truth[keep])) < 2:
        return float("nan")
    return float(roc_auc_score((truth[keep] == ENTAILMENT).astype(int), score[keep]))


def tfidf_baseline(train, test, max_features: int = 200_000) -> dict:
    """Word unigrams and bigrams of both sentences, concatenated, into a linear model."""

    def join(split):
        return [f"{o} [SEP] {s}" for o, s in zip(split["original"], split["simplification"])]

    vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True)
    matrix = vectoriser.fit_transform(join(train))
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(matrix, labels(train))
    truth = labels(test)
    probabilities = model.predict_proba(vectoriser.transform(join(test)))
    predicted = probabilities.argmax(axis=1)
    margin = probabilities[:, ENTAILMENT] - probabilities[:, CONTRADICTION]
    return {
        "features": int(matrix.shape[1]),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": macro_f1(truth, predicted),
        "auc_entailment_vs_contradiction": auc_entail_vs_contra(margin, truth),
    }


@click.command()
@click.option("--corpus", default="datastore/polarity", show_default=True)
@click.option("--json-out", default=None)
def main(corpus: str, json_out: Optional[str]) -> None:
    """Run the three baselines and print what each one bounds."""
    splits = load_from_disk(f"{corpus}/corpus")
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    click.echo(f"train {len(train)}  dev {len(dev)}  test {len(test)}")

    findings = {
        "majority": majority_baseline(train, test),
        "overlap": overlap_baseline(train, dev, test),
        "tfidf_logreg": tfidf_baseline(train, test),
        "n_test": len(test),
    }
    for name in ("majority", "overlap", "tfidf_logreg"):
        got = findings[name]
        auc = got.get("auc_entailment_vs_contradiction")
        extra = f"  AUC {auc:.3f}" if auc is not None else ""
        click.echo(f"{name:14s} accuracy {got['accuracy']:.4f}  macro-F1 {got['macro_f1']:.4f}{extra}")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(findings, handle, indent=2)
        click.echo(f"brut : {json_out}")


if __name__ == "__main__":
    main()
