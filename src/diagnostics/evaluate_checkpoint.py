"""Score an existing checkpoint on a v2 corpus variant, without retraining anything.

Why this exists. The grid measures models we trained. It says nothing about the model
already published as ``davebulaval/MeaningBERT``, which is what everyone downloads today
and what v2 has to beat to be worth publishing. That comparison needs the published
weights run through the same test set and the same metrics as the grid, not a number
copied from the v1 paper, which was measured on a different split.

The output JSON has the same shape as a training run's, so it lands in ``results/grid``
and every existing table, figure and aggregation picks it up with no special case.

One caveat travels with every v1 number this produces, and it must be stated wherever the
number is: the published model was trained on CSMD with a ROW-level split, while the c
rung is the same corpus re-split by source sentence. Part of this test set was therefore
in its training data. Its score here is an upper bound, not a fair measurement, and the
comparison is only conclusive in one direction: if v1 loses while advantaged, it loses.

Run::

    PYTHONPATH=src python src/diagnostics/evaluate_checkpoint.py \\
        --checkpoint davebulaval/MeaningBERT --variant-path data/v2/c_none \\
        --results-json results/grid/published-v1/c_none.json
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import click
import numpy as np
from datasets import load_from_disk
from scipy import stats

try:  # PYTHONPATH=src.
    from meaningbert.scorer import MeaningBERTScorer
except ImportError:  # pragma: no cover
    from src.meaningbert.scorer import MeaningBERTScorer  # type: ignore


def metrics(gold: np.ndarray, pred: np.ndarray, prefix: str) -> dict:
    """The grid's metrics, computed the same way, so the rows are comparable."""
    err = pred - gold
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((gold - gold.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    if len(gold) > 1 and gold.std() > 0 and pred.std() > 0:
        r, p = stats.pearsonr(gold, pred)
    else:
        r, p = float("nan"), float("nan")
    return {
        f"{prefix}_loss": float(np.mean((err / 100.0) ** 2)),
        f"{prefix}_rmse": rmse,
        f"{prefix}_R2": round(r2, 3) if math.isfinite(r2) else r2,
        f"{prefix}_pearson_corr": float(r),
        f"{prefix}_pearson_pvalue": float(p),
        f"{prefix}_mean_score": float(pred.mean()),
        f"{prefix}_st_dev_score": float(pred.std(ddof=1)) if len(pred) > 1 else 0.0,
        f"{prefix}_diverged": 0.0,
    }


def sanity(scorer: MeaningBERTScorer, rows, kind: str) -> dict:
    """Identical pairs must score 100, unrelated pairs must score 0.

    Reported as a mean AND as a threshold ratio, because the ratio alone hides the
    distance: "38 percent above 95" says nothing about whether the rest sit at 94 or 60.
    """
    if len(rows) == 0:
        return {}
    pred = np.array(scorer.score(rows["original"], rows["simplification"]), dtype=float)
    key = "identical_sentences" if kind == "identical" else "unrelated_sentences"
    gold = np.full(len(pred), 100.0 if kind == "identical" else 0.0)
    out = {
        f"test/{key}_rmse": float(np.sqrt(np.mean((pred - gold) ** 2))),
        f"test/{key}_mean_score": float(pred.mean()),
        f"test/{key}_st_dev_score": float(pred.std(ddof=1)) if len(pred) > 1 else 0.0,
        f"test/{key}_diverged": 0.0,
    }
    if kind == "identical":
        out[f"test/{key}_ratio_95"] = float(100.0 * np.mean(pred > 95.0))
        out[f"test/{key}_ratio_equals"] = float(100.0 * np.mean(pred >= 99.999))
    else:
        out[f"test/{key}_ratio_5"] = float(100.0 * np.mean(pred < 5.0))
        out[f"test/{key}_ratio_equals"] = float(100.0 * np.mean(pred <= 0.001))
    return out


@click.command()
@click.option("--checkpoint", required=True, help="HuggingFace id or local path.")
@click.option("--variant-path", required=True, help="A v2 corpus variant, e.g. data/v2/c_none.")
@click.option("--results-json", required=True, help="Where the run-shaped JSON goes.")
@click.option("--batch-size", default=32, show_default=True)
@click.option("--max-length", default=256, show_default=True, help="Same bound the grid trains under.")
def main(checkpoint: str, variant_path: str, results_json: str, batch_size: int, max_length: int) -> None:
    """Score *checkpoint* on *variant_path* and write a run-shaped JSON."""
    data = load_from_disk(variant_path)
    scorer = MeaningBERTScorer(checkpoint, batch_size=batch_size)
    scorer.max_length = min(scorer.max_length, max_length)
    print(f"checkpoint  : {checkpoint}")
    print(f"tete lue    : {scorer.head}")
    print(f"variante    : {variant_path}  (test={len(data['test'])})")

    test = data["test"]
    pred = np.array(scorer.score(test["original"], test["simplification"]), dtype=float)
    gold = np.array(test["label"], dtype=float)
    payload = {
        "run_name": f"evaluated_{os.path.basename(checkpoint)}_{os.path.basename(variant_path)}",
        "checkpoint": checkpoint,
        "variant_path": os.path.abspath(variant_path),
        "output_head": scorer.head,
        "seed": -1,
        "num_epochs": 0,
        "epochs_trained": 0.0,
        "evaluation_only": True,
        "rows": {split: len(data[split]) for split in data},
        "test": metrics(gold, pred, "test"),
    }
    # Ventilation par corpus. Elle existe pour une raison precise : le modele publie a ete
    # entraine sur CSMD, donc toute ligne de test venant de CSMD a pu etre vue. Les lignes
    # des corpus ajoutes en v2 ne l'ont pas ete, et elles seules donnent une comparaison
    # que la fuite ne gonfle pas.
    corpora = np.array(test["corpus"])
    by_corpus = {}
    for name in sorted(set(corpora.tolist())):
        mask = corpora == name
        if mask.sum() < 3:
            continue
        m = metrics(gold[mask], pred[mask], "test")
        by_corpus[name] = {"n": int(mask.sum()), "pearson": m["test_pearson_corr"],
                           "rmse": m["test_rmse"], "R2": m["test_R2"]}
    payload["by_corpus"] = by_corpus

    if "sanity" in data:
        s = data["sanity"]
        payload["identical"] = sanity(scorer, s.filter(lambda r: r["source"] == "identical"), "identical")
        payload["unrelated"] = sanity(scorer, s.filter(lambda r: r["source"] == "unrelated"), "unrelated")

    os.makedirs(os.path.dirname(os.path.abspath(results_json)) or ".", exist_ok=True)
    with open(results_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    t = payload["test"]
    print(f"\nPearson {t['test_pearson_corr']:.3f}   RMSE {t['test_rmse']:.2f}   R2 {t['test_R2']}")
    print(f"moyenne predite {t['test_mean_score']:.2f} (etiquettes {gold.mean():.2f}), "
          f"ecart-type {t['test_st_dev_score']:.2f} (etiquettes {gold.std(ddof=1):.2f})")
    if payload.get("identical"):
        i, u = payload["identical"], payload["unrelated"]
        print(f"identiques : moyenne {i['test/identical_sentences_mean_score']:.2f}, "
              f"{i['test/identical_sentences_ratio_95']:.1f} % au-dessus de 95")
        print(f"non reliees: moyenne {u['test/unrelated_sentences_mean_score']:.2f}, "
              f"{u['test/unrelated_sentences_ratio_5']:.1f} % au-dessous de 5")
    if payload.get("by_corpus"):
        print("\npar corpus :")
        for name, m in payload["by_corpus"].items():
            print(f"  {name:14} n={m['n']:5d}  Pearson {m['pearson']:.3f}  RMSE {m['rmse']:6.2f}")
    print(f"\necrit dans {results_json}")


if __name__ == "__main__":
    main()
