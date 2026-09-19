"""Read the eight runs of the v2 experiment and decompose where the gain comes from.

The ladder is designed so each step isolates one thing. Reading the table top to bottom:

* ``a -> b`` is the source-sentence leak (H5). Expect the score to **drop**: condition
  ``a`` is measured on a test set whose sentences the model has seen.
* ``b -> c`` is the permuted labels (H6). Expect it to **rise**, because 26.6 percent of
  CSMD's annotated rows were carrying another pair's label.
* ``c -> d`` is what the three added corpora are actually worth.
* ``none -> full`` is the augmentation, measured for the first time against a real
  no-augmentation baseline. The v1 sweep never ran one.

Reported alongside every score: the predicted mean and spread, because
``docs/H1-diagnostic-calibration.md`` showed an RMSE that was mostly amplitude compression
and invisible without them.

Run::

    PYTHONPATH=src python src/figures_generator/analyze_v2_experiment.py --runs-dir results/v2-runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Optional

CONDITION_LABELS: dict[str, str] = {
    "a": "v1 corpus, v1 row split",
    "b": "v1 corpus, grouped split",
    "c": "v1 corrected (H6), grouped",
    "d": "v2 corpus, grouped",
}
MODE_LABELS: dict[str, str] = {"none": "no augmentation", "full": "swap + BT + generated"}

#: Spread of the merged CSMD labels, used to turn a correlation into the RMSE floor.
LABEL_STD: float = 37.01


@dataclass
class RunResult:
    """One finished run."""

    variant: str
    condition: str
    mode: str
    pearson: float
    rmse: float
    r2: float
    pred_mean: float
    pred_std: float
    identical_ratio_95: float
    unrelated_ratio_5: float
    epochs: float
    train_rows: int
    diverged: bool

    @property
    def rmse_floor(self) -> float:
        """RMSE the best affine rescaling of these predictions would reach."""
        if not math.isfinite(self.pearson):
            return float("nan")
        return LABEL_STD * math.sqrt(max(0.0, 1.0 - min(1.0, abs(self.pearson)) ** 2))


def _number(mapping: dict, *keys: str) -> float:
    for key in keys:
        value = mapping.get(key)
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        return number
    return float("nan")


def load_run(path: str) -> Optional[RunResult]:
    """Read one run's JSON, or ``None`` if it is unreadable."""
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    variant = os.path.splitext(os.path.basename(path))[0]
    condition, _, mode = variant.partition("_")
    test = payload.get("test", {}) or {}
    identical = payload.get("identical", {}) or {}
    unrelated = payload.get("unrelated", {}) or {}

    return RunResult(
        variant=variant,
        condition=condition,
        mode=mode,
        pearson=_number(test, "test_pearson_corr", "test/pearson_corr"),
        rmse=_number(test, "test_rmse", "test/rmse"),
        r2=_number(test, "test_R2", "test/R2"),
        pred_mean=_number(test, "test_mean_score", "test/mean_score"),
        pred_std=_number(test, "test_st_dev_score", "test/st_dev_score"),
        identical_ratio_95=_number(
            identical, "test/identical_sentences_ratio_95", "test/identical_sentences_ratio_95".replace("/", "_")
        ),
        unrelated_ratio_5=_number(
            unrelated, "test/unrelated_sentences_ratio_5", "test/unrelated_sentences_ratio_5".replace("/", "_")
        ),
        epochs=_number(payload, "epochs_trained"),
        train_rows=int(_number(payload.get("rows", {}), "train") or 0),
        diverged=bool(_number(test, "test_diverged", "test/diverged") == 1.0),
    )


def load_runs(runs_dir: str) -> list[RunResult]:
    """Read every run JSON in *runs_dir*, ordered by condition then mode."""
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        if name.endswith(".json"):
            run = load_run(os.path.join(runs_dir, name))
            if run is not None:
                runs.append(run)
    order = {"none": 0, "full": 1}
    return sorted(runs, key=lambda r: (r.condition, order.get(r.mode, 9)))


def _fmt(value: float, width: int = 7, digits: int = 3) -> str:
    return f"{'-':>{width}}" if not math.isfinite(value) else f"{value:>{width}.{digits}f}"


def render(runs: list[RunResult]) -> str:
    """Render the results table and the decomposition of the gains."""
    lines = [
        f"{'variant':10} {'condition':26} {'augment.':22} {'Pearson':>8} {'RMSE':>7} {'floor':>7} "
        f"{'R2':>7} {'pred_mu':>8} {'pred_sd':>8} {'ident>95':>9} {'unrel<5':>8} {'train':>7} {'ep':>5}",
        "-" * 150,
    ]
    for run in runs:
        flag = "  DIVERGED" if run.diverged else ""
        lines.append(
            f"{run.variant:10} {CONDITION_LABELS.get(run.condition, '?'):26} "
            f"{MODE_LABELS.get(run.mode, run.mode):22} {_fmt(run.pearson, 8)} {_fmt(run.rmse, 7, 2)} "
            f"{_fmt(run.rmse_floor, 7, 2)} {_fmt(run.r2, 7)} {_fmt(run.pred_mean, 8, 2)} "
            f"{_fmt(run.pred_std, 8, 2)} {_fmt(run.identical_ratio_95, 9, 1)} {_fmt(run.unrelated_ratio_5, 8, 1)} "
            f"{run.train_rows:>7} {_fmt(run.epochs, 5, 0)}{flag}"
        )

    by_variant = {run.variant: run for run in runs}

    lines += ["", "Decomposition, Pearson on the test split", "-" * 60]
    for mode in ("none", "full"):
        steps = [
            ("a -> b", "source-sentence leak (H5)", f"a_{mode}", f"b_{mode}"),
            ("b -> c", "permuted labels (H6)", f"b_{mode}", f"c_{mode}"),
            ("c -> d", "the three added corpora", f"c_{mode}", f"d_{mode}"),
        ]
        lines.append(f"  {MODE_LABELS[mode]}:")
        for arrow, what, left, right in steps:
            if left in by_variant and right in by_variant:
                delta = by_variant[right].pearson - by_variant[left].pearson
                lines.append(f"    {arrow}  {what:32} {delta:+.3f}")

    lines += ["", "Augmentation, at each rung", "-" * 60]
    for condition in ("a", "b", "c", "d"):
        none, full = by_variant.get(f"{condition}_none"), by_variant.get(f"{condition}_full")
        if none and full:
            lines.append(
                f"  {condition}  {CONDITION_LABELS[condition]:26} "
                f"Pearson {full.pearson - none.pearson:+.3f}   RMSE {full.rmse - none.rmse:+.2f}"
            )

    target = by_variant.get("d_full") or by_variant.get("d_none")
    if target:
        lines += [
            "",
            "Against the PRODUIT.md target",
            "-" * 60,
            f"  best v2 run   : {target.variant}, Pearson {target.pearson:.3f}, RMSE {target.rmse:.2f}",
            "  target        : Pearson >= 0.914, RMSE < 15",
            f"  RMSE floor at this correlation: {target.rmse_floor:.2f}",
        ]
        if math.isfinite(target.rmse) and math.isfinite(target.rmse_floor):
            gap = target.rmse - target.rmse_floor
            lines.append(f"  still recoverable by recalibration alone: {gap:.2f} RMSE points")
    return "\n".join(lines)


def main() -> None:
    """Print the v2 experiment table and the decomposition of its gains."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="results/v2-runs", help="Directory holding one JSON per run.")
    parser.add_argument("--markdown-out", default=None, help="Optional path for the rendered table.")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"No run found in {args.runs_dir}")
        return

    report = render(runs)
    print(report)
    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as handle:
            handle.write("# Resultats de l'experience v2\n\n```\n" + report + "\n```\n")
        print(f"\nwritten to {args.markdown_out}")


if __name__ == "__main__":
    main()
