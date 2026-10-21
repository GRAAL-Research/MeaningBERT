"""v4 hypothesis V4-H3: can the French magnitude be borrowed?

The signed score multiplies a magnitude by ``1 - 2 p_contra``. The v4 framing keeps the
polarity head as the thing to build and borrows the magnitude, provided a borrowed one
tracks human judgement in French at least as well as multilingual BERTScore
(``docs/produit/cadrage-v4-multilingue.md``, criteria). FrJUDGE (Beauchemin et al., 2025)
is the only French corpus of human meaning-preservation ratings; its test split was never
seen by JUDGEBERT, and no candidate here was trained on any of it.

Candidates, each reading French directly or its English translation:

* ``MeaningBERT`` v1 and v2-large, the magnitude of the v3 composition;
* BERTScore F1 on ``xlm-roberta-large``, the bar.

Run::

    PYTHONPATH=src python src/diagnostics/magnitude_fr.py --out results/v4/magnitude-frjudge.json
"""

from __future__ import annotations

import glob
import json
import os
import re
import urllib.request
from typing import Any, Final

import click
import numpy as np
from scipy.stats import pearsonr, spearmanr

try:
    from diagnostics.significance import holm, williams
except ImportError:  # pragma: no cover - run from inside ``src/diagnostics``
    from significance import holm, williams  # type: ignore

FRJUDGE_URL: Final[str] = (
    "https://raw.githubusercontent.com/GRAAL-Research/JUDGEBERT/HEAD/datastore/training/{split}_FrJUDGE.jsonl"
)

#: (name, checkpoint, subfolder) of the MeaningBERT magnitudes.
MEANINGBERT: Final[list[tuple[str, str, str]]] = [
    ("MeaningBERT v1", "davebulaval/MeaningBERT", ""),
    ("MeaningBERT v2-large", "davebulaval/MeaningBERT", "large"),
]

BERTSCORE_MODEL: Final[str] = "xlm-roberta-large"
#: Layer used by the bert_score package for xlm-roberta-large.
BERTSCORE_LAYER: Final[int] = 17


#: The bar every candidate magnitude is tested against.
BAR: Final[str] = "BERTScore XLM-R-large / français"


def trained_models(root: str) -> dict[str, list[str]]:
    """Magnitude heads trained by ``run_magnitude_queue.sh``, grouped by family.

    The family is the folder under *root* (``mdeberta-fr``: backbone and training corpus);
    each holds one ``..._seed<N>`` checkpoint per seed, averaged into one candidate.
    """
    families: dict[str, list[str]] = {}
    for config in sorted(glob.glob(os.path.join(root, "*", "*_seed*", "config.json"))):
        folder = os.path.dirname(config)
        if re.search(r"_seed\d+$", folder):
            families.setdefault(os.path.basename(os.path.dirname(folder)), []).append(folder)
    return families


def compare(scores: dict[str, np.ndarray], gold: np.ndarray, reference: str = BAR) -> dict[str, dict[str, float]]:
    """Each candidate against *reference* on the same pairs: Williams' t, Holm over all of them."""
    raw: dict[str, dict[str, float]] = {}
    for name, values in scores.items():
        if name == reference:
            continue
        r_a, r_b = pearsonr(values, gold)[0], pearsonr(scores[reference], gold)[0]
        test = williams(float(r_a), float(r_b), float(pearsonr(values, scores[reference])[0]), len(gold))
        raw[name] = {"diff": float(r_a - r_b), **test}
    adjusted = holm({name: block["p"] for name, block in raw.items() if not np.isnan(block["p"])})
    for name, block in raw.items():
        block["p_holm"] = adjusted.get(name, float("nan"))
    return raw


def markdown(result: dict[str, Any]) -> str:
    """The magnitude benchmark as one table, best Pearson first."""

    def p_text(value: float) -> str:
        return "-" if np.isnan(value) else ("< 0.01" if value < 0.01 else f"{value:.2f}")

    lines = [
        "## Magnitude sur FrJUDGE (Pearson avec le jugement humain)",
        "",
        "| Candidat | Pearson (toutes) | IC 95 % | Pearson (test) | Spearman | Δ BERTScore | p Williams |",
        "|---|---|---|---|---|---|---|",
    ]
    candidates = result["candidates"]
    for name in sorted(candidates, key=lambda n: -candidates[n]["all"]["pearson"]):
        block = candidates[name]["all"]
        versus = result["versus_bar"].get(name)
        delta = f"{versus['diff']:+.2f}" if versus else "réf."
        lines.append(
            f"| {name} | {block['pearson']:.2f} | [{block['pearson_low']:.2f}, {block['pearson_high']:.2f}] "
            f"| {candidates[name]['test']['pearson']:.2f} | {block['spearman']:.2f} | {delta} "
            f"| {p_text(versus['p_holm']) if versus else ''} |"
        )
    lines += ["", "p : test de Williams apparié contre BERTScore, ajusté par Holm.", ""]
    return "\n".join(lines)


def load_frjudge(split: str) -> list[dict[str, Any]]:
    """One FrJUDGE split as ``original``, ``simplification``, ``label`` rows."""
    with urllib.request.urlopen(FRJUDGE_URL.format(split=split), timeout=60) as response:  # noqa: S310
        lines = response.read().decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def correlation(predicted: np.ndarray, gold: np.ndarray, rounds: int = 2000, seed: int = 42) -> dict[str, float]:
    """Pearson and Spearman, with a 95 % bootstrap interval on Pearson."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(rounds):
        index = rng.integers(0, len(gold), len(gold))
        if np.std(predicted[index]) > 0 and np.std(gold[index]) > 0:
            draws.append(pearsonr(predicted[index], gold[index])[0])
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "n": int(len(gold)),
        "pearson": float(pearsonr(predicted, gold)[0]),
        "pearson_low": float(low),
        "pearson_high": float(high),
        "spearman": float(spearmanr(predicted, gold)[0]),
    }


def greedy_f1(candidate: np.ndarray, reference: np.ndarray) -> float:
    """BERTScore F1 from two token-embedding matrices (no idf, no baseline rescaling)."""
    candidate = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    similarity = candidate @ reference.T
    precision = similarity.max(axis=1).mean()
    recall = similarity.max(axis=0).mean()
    if precision + recall <= 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def bertscore(left: list[str], right: list[str]) -> np.ndarray:
    """BERTScore F1 on :data:`BERTSCORE_MODEL`, special tokens excluded."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BERTSCORE_MODEL)
    model = AutoModel.from_pretrained(BERTSCORE_MODEL).to(device).eval()

    def embed(text: str) -> np.ndarray:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            states = model(**encoded, output_hidden_states=True).hidden_states[BERTSCORE_LAYER][0]
        return states[1:-1].float().cpu().numpy()

    return np.array([greedy_f1(embed(a), embed(b)) for a, b in zip(left, right)])


def meaningbert(checkpoint: str, subfolder: str, left: list[str], right: list[str]) -> np.ndarray:
    """0-100 MeaningBERT score for each pair."""
    try:
        from diagnostics.cross_task import Model
    except ImportError:  # pragma: no cover - run from inside ``src/diagnostics``
        from cross_task import Model  # type: ignore

    model = Model(checkpoint, subfolder or None, batch_size=8)
    return model.meaning_score(model.logits(left, right))


@click.command()
@click.option("--out", default="results/v4/magnitude-frjudge.json", show_default=True)
@click.option("--translations", default="results/v4/discovery/translations-frjudge.json", show_default=True)
@click.option("--trained-root", default="results/magnitude-v4/models", show_default=True)
def main(out: str, translations: str, trained_root: str) -> None:
    """Correlation de chaque magnitude candidate avec le jugement humain sur FrJUDGE."""
    try:
        from diagnostics.discovery_fr import translate
    except ImportError:  # pragma: no cover
        from discovery_fr import translate  # type: ignore

    rows = {split: load_frjudge(split) for split in ("train", "test")}
    rows["all"] = rows["train"] + rows["test"]
    every = rows["all"]
    table = translate([s for r in every for s in (r["original"], r["simplification"])], translations)

    left_fr = [r["original"] for r in every]
    right_fr = [r["simplification"] for r in every]
    left_en = [table[s] for s in left_fr]
    right_en = [table[s] for s in right_fr]

    scores: dict[str, np.ndarray] = {BAR: bertscore(left_fr, right_fr)}
    for name, checkpoint, subfolder in MEANINGBERT:
        scores[f"{name} / français"] = meaningbert(checkpoint, subfolder, left_fr, right_fr)
        scores[f"{name} / traduit"] = meaningbert(checkpoint, subfolder, left_en, right_en)
    # The v4 multilingual MeaningBERT, one candidate per backbone and training corpus, its
    # seeds averaged; read in French and, for comparison, in translation.
    for family, folders in trained_models(trained_root).items():
        for label, left, right in (("français", left_fr, right_fr), ("traduit", left_en, right_en)):
            stacked = [meaningbert(folder, "", left, right) for folder in folders]
            scores[f"MeaningBERT v4 {family} ({len(folders)} graines) / {label}"] = np.mean(stacked, axis=0)

    gold = np.array([float(r["label"]) for r in every])
    n_train = len(rows["train"])
    result: dict[str, Any] = {"n": {split: len(r) for split, r in rows.items()}, "candidates": {}}
    for name, values in scores.items():
        result["candidates"][name] = {
            "all": correlation(values, gold),
            "test": correlation(values[n_train:], gold[n_train:]),
        }
    result["versus_bar"] = compare(scores, gold)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    with open(os.path.splitext(out)[0] + ".md", "w", encoding="utf-8") as handle:
        handle.write(markdown(result))
    for name, block in result["candidates"].items():
        click.echo(f"{name:40s} Pearson {block['all']['pearson']:.2f}  (test {block['test']['pearson']:.2f})")


if __name__ == "__main__":
    main()
