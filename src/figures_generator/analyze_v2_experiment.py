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

    arch: str
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
    def objective(self) -> float:
        """The stated goal of v2, as one number: maximise Pearson with both sanity checks near 100.

        A product, not a sum. A model that correlates beautifully but scores identical
        pairs at 90 has not met the goal, and an average would hide that; multiplying makes
        any one weak term drag the whole score down. Read it next to its three components,
        never instead of them.
        """
        terms = (self.pearson, self.identical_ratio_95 / 100.0, self.unrelated_ratio_5 / 100.0)
        if any(not math.isfinite(term) for term in terms):
            return float("nan")
        return float(terms[0] * terms[1] * terms[2])

    @property
    def rmse_floor(self) -> float:
        """RMSE the best affine rescaling of these predictions would reach."""
        if not math.isfinite(self.pearson):
            return float("nan")
        return LABEL_STD * math.sqrt(max(0.0, 1.0 - min(1.0, abs(self.pearson)) ** 2))


def _int_or_zero(value: float) -> int:
    """Integer of *value*, or 0 when it is NaN. ``int(nan)`` raises, and a crash in the
    reporting layer over a missing row count would hide every result that did survive."""
    return 0 if not math.isfinite(value) else int(value)


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

    # results/ also holds diagnostics: host probes, the calibration audit, the leakage
    # audit. They are JSON and they are not runs. A run is identified by carrying test
    # metrics, not by living in the right directory.
    if not isinstance(payload, dict) or not isinstance(payload.get("test"), dict):
        return None

    variant = os.path.splitext(os.path.basename(path))[0]
    # Grid layout is results/grid/<arch>/<variant>.json; the flat layout has no arch level.
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    arch = payload.get("checkpoint", "").split("/")[-1] or parent
    condition, _, mode = variant.partition("_")
    test = payload.get("test", {}) or {}
    identical = payload.get("identical", {}) or {}
    unrelated = payload.get("unrelated", {}) or {}

    return RunResult(
        arch=arch,
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
        train_rows=_int_or_zero(_number(payload.get("rows", {}), "train")),
        diverged=bool(_number(test, "test_diverged", "test/diverged") == 1.0),
    )


def load_runs(runs_dir: str) -> list[RunResult]:
    """Read every run JSON under *runs_dir*, flat or one directory per architecture."""
    runs = []
    for root, _, names in os.walk(runs_dir):
        for name in sorted(names):
            if name.endswith(".json"):
                run = load_run(os.path.join(root, name))
                if run is not None:
                    runs.append(run)
    order = {"none": 0, "full": 1}
    return sorted(runs, key=lambda r: (r.arch, r.condition, order.get(r.mode, 9)))


def _fmt(value: float, width: int = 7, digits: int = 3) -> str:
    return f"{'-':>{width}}" if not math.isfinite(value) else f"{value:>{width}.{digits}f}"


def render(runs: list[RunResult]) -> str:
    """Render the results, ranked by the stated objective, plus the diagnostic ladder."""
    lines = [
        "Objectif : maximiser Pearson avec identiques et non reliees au plus pres de 100 %.",
        "objectif = Pearson x (identiques>95) x (non reliees<5), en produit pour qu'un",
        "seul terme faible tire tout vers le bas.",
        "",
        f"{'arch':22} {'variant':8} {'objectif':>9} {'Pearson':>8} {'ident>95':>9} {'unrel<5':>8} "
        f"{'RMSE':>7} {'floor':>7} {'R2':>7} {'pred_mu':>8} {'pred_sd':>8} {'train':>7} {'ep':>4}",
        "-" * 140,
    ]
    for run in sorted(runs, key=lambda r: (-(r.objective if math.isfinite(r.objective) else -1))):
        flag = "  DIVERGED" if run.diverged else ""
        lines.append(
            f"{run.arch[:22]:22} {run.variant:8} {_fmt(run.objective, 9)} {_fmt(run.pearson, 8)} "
            f"{_fmt(run.identical_ratio_95, 9, 1)} {_fmt(run.unrelated_ratio_5, 8, 1)} "
            f"{_fmt(run.rmse, 7, 2)} {_fmt(run.rmse_floor, 7, 2)} {_fmt(run.r2, 7)} "
            f"{_fmt(run.pred_mean, 8, 2)} {_fmt(run.pred_std, 8, 2)} {run.train_rows:>7} {_fmt(run.epochs, 4, 0)}{flag}"
        )

    by_arch: dict[str, dict[str, RunResult]] = {}
    for run in runs:
        by_arch.setdefault(run.arch, {})[run.variant] = run

    lines += ["", "Corpus : v1 corrige (c) contre v2 (d), a augmentation egale", "-" * 72]
    for arch, variants in sorted(by_arch.items()):
        for mode in ("none", "full"):
            left, right = variants.get(f"c_{mode}"), variants.get(f"d_{mode}")
            if left and right:
                lines.append(
                    f"  {arch[:22]:22} {MODE_LABELS[mode]:22} Pearson {right.pearson - left.pearson:+.3f}"
                    f"   objectif {right.objective - left.objective:+.3f}"
                )

    lines += ["", "Augmentation : aucune contre les trois ensemble, a corpus egal", "-" * 72]
    for arch, variants in sorted(by_arch.items()):
        for condition in ("c", "d"):
            none, full = variants.get(f"{condition}_none"), variants.get(f"{condition}_full")
            if none and full:
                label = CONDITION_LABELS[condition]
                lines.append(
                    f"  {arch[:22]:22} {label:26} Pearson {full.pearson - none.pearson:+.3f}"
                    f"   ident {full.identical_ratio_95 - none.identical_ratio_95:+.1f} pts"
                    f"   objectif {full.objective - none.objective:+.3f}"
                )

    lines += ["", "Diagnostics, sur l'architecture de reference seulement", "-" * 72]
    for arch, variants in sorted(by_arch.items()):
        for arrow, what, left_key, right_key in (
            ("a -> b", "fuite par phrase source (H5)", "a", "b"),
            ("b -> c", "etiquettes permutees (H6)", "b", "c"),
        ):
            for mode in ("none", "full"):
                left, right = variants.get(f"{left_key}_{mode}"), variants.get(f"{right_key}_{mode}")
                if left and right:
                    lines.append(
                        f"  {arch[:22]:22} {arrow}  {what:30} {MODE_LABELS[mode][:12]:12} "
                        f"Pearson {right.pearson - left.pearson:+.3f}"
                        f"   ident {right.identical_ratio_95 - left.identical_ratio_95:+.1f} pts"
                    )

    finite = [r for r in runs if math.isfinite(r.objective)]
    if finite:
        best = max(finite, key=lambda r: r.objective)
        lines += [
            "",
            "Meilleure configuration au sens de l'objectif",
            "-" * 72,
            f"  {best.arch} / {best.variant}",
            f"  Pearson {best.pearson:.3f}   identiques>95 {best.identical_ratio_95:.1f} %   "
            f"non reliees<5 {best.unrelated_ratio_5:.1f} %   RMSE {best.rmse:.2f}",
            "  cible PRODUIT.md : Pearson >= 0,914, RMSE < 15",
            f"  plancher de RMSE a cette correlation : {best.rmse_floor:.2f}",
        ]
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
