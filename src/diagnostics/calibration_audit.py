"""Audit the checkpoint sweep for the three calibration failures behind hypothesis H1.

See ``PRODUIT.md``. The sweep reports Pearson around 0.79 with an RMSE of 35 to 54 on a
0-100 scale and an ``Identical =100%`` ratio of exactly zero. This script decides whether
that comes from the data, the model, or the output layer.

It checks three things per run:

1. **Collapse.** A run whose predicted standard deviation is ~0 has degenerated to a
   constant output. ``metrics._sanitize_predictions`` turns the resulting NaN into 0.0,
   so a dead run is reported as a model that predicts zero everywhere. That is a
   plausible-looking number, which is why it survives into the aggregate.
2. **Amplitude compression.** Comparing the predicted mean and spread to the label mean
   and spread shows whether the model learned the ranking but not the scale.
3. **Recoverable error.** The RMSE an optimal affine rescaling would reach is
   ``sigma_y * sqrt(1 - r^2)``. The gap between that and the reported RMSE is pure
   calibration loss, recoverable without touching the backbone or adding a single row.

Run::

    python src/diagnostics/calibration_audit.py --entity davebulaval \
        --project meaningbert-checkpoint-sweep
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# Labels of the merged CSMD corpus (base 1355 + identical 359 + unrelated 359).
# Recomputed by ``--recompute-labels``; hard-coded so the audit runs offline.
DEFAULT_LABEL_MEAN: float = 62.66
DEFAULT_LABEL_STD: float = 37.01

# A run whose predictions vary less than this has collapsed to a constant.
COLLAPSE_STD_THRESHOLD: float = 1e-3


@dataclass
class RunAudit:
    """Verdict for a single sweep run."""

    name: str
    checkpoint: str
    augmentation: str
    rmse: float
    pearson: float
    pred_mean: float
    pred_std: float
    collapsed: bool
    #: RMSE reachable by the best affine rescaling of the same predictions.
    rmse_after_affine: float
    #: Share of the reported RMSE that is pure calibration loss.
    recoverable_fraction: float


@dataclass
class AuditReport:
    """Aggregate verdict over a sweep."""

    label_mean: float
    label_std: float
    n_runs: int
    n_collapsed: int
    runs: list[RunAudit] = field(default_factory=list)

    def by_checkpoint(self) -> dict[str, dict[str, float]]:
        """Median metrics per checkpoint, collapsed runs excluded."""
        grouped: dict[str, list[RunAudit]] = collections.defaultdict(list)
        for run in self.runs:
            if not run.collapsed:
                grouped[run.checkpoint].append(run)
        return {
            checkpoint: {
                "n": len(audits),
                "pearson": statistics.median(a.pearson for a in audits),
                "rmse": statistics.median(a.rmse for a in audits),
                "rmse_after_affine": statistics.median(a.rmse_after_affine for a in audits),
                "pred_mean": statistics.median(a.pred_mean for a in audits),
                "pred_std": statistics.median(a.pred_std for a in audits),
            }
            for checkpoint, audits in sorted(grouped.items())
        }


def optimal_affine_rmse(label_std: float, pearson: float) -> float:
    """RMSE left after the best affine rescaling ``a * p + b`` of the predictions.

    The optimum of ``E[(y - a*p - b)^2]`` over ``a`` and ``b`` is ``var(y) * (1 - r^2)``,
    so only the part of the error that the correlation cannot explain survives. Everything
    else is a location and scale mismatch, which is a calibration problem.

    Args:
        label_std: Standard deviation of the gold labels.
        pearson: Pearson correlation between predictions and labels.

    Returns:
        The post-rescaling RMSE. Returns ``label_std`` when *pearson* is not usable, since
        a prediction uncorrelated with the labels cannot beat predicting their mean.
    """
    if not math.isfinite(pearson):
        return label_std
    clamped = max(-1.0, min(1.0, pearson))
    return label_std * math.sqrt(max(0.0, 1.0 - clamped**2))


def audit_run(
    name: str,
    checkpoint: str,
    augmentation: str,
    summary: dict[str, Any],
    label_std: float,
) -> Optional[RunAudit]:
    """Turn one run summary into a :class:`RunAudit`, or ``None`` if it has no test metrics."""

    def number(key: str) -> float:
        value = summary.get(key)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")

    rmse = number("test/rmse")
    if not math.isfinite(rmse):
        return None

    pred_std = number("test/st_dev_score")
    pearson = number("test/pearson_corr")
    # A usable run needs a finite, non-degenerate spread and a defined correlation.
    # A non-finite std is not "unknown", it is what a dead run reports.
    collapsed = not math.isfinite(pred_std) or pred_std < COLLAPSE_STD_THRESHOLD or not math.isfinite(pearson)

    floor = optimal_affine_rmse(label_std, pearson)
    recoverable = 0.0 if rmse <= 0 or not math.isfinite(floor) else max(0.0, (rmse - floor) / rmse)

    return RunAudit(
        name=name,
        checkpoint=checkpoint,
        augmentation=augmentation,
        rmse=rmse,
        pearson=pearson,
        pred_mean=number("test/mean_score"),
        pred_std=pred_std,
        collapsed=collapsed,
        rmse_after_affine=floor,
        recoverable_fraction=recoverable,
    )


def audit_runs(records: Iterable[tuple[str, str, str, dict[str, Any]]], label_mean: float, label_std: float) -> AuditReport:
    """Audit an iterable of ``(name, checkpoint, augmentation, summary)`` records."""
    audits = [a for a in (audit_run(*record, label_std=label_std) for record in records) if a is not None]
    return AuditReport(
        label_mean=label_mean,
        label_std=label_std,
        n_runs=len(audits),
        n_collapsed=sum(1 for a in audits if a.collapsed),
        runs=audits,
    )


def _fetch_from_wandb(entity: str, project: str) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Pull ``(name, checkpoint, augmentation, summary)`` for every finished run."""
    import wandb  # imported lazily so the module stays importable offline

    api = wandb.Api(timeout=90)
    records = []
    for run in api.runs(f"{entity}/{project}", per_page=200):
        if run.state != "finished":
            continue
        config = dict(run.config)
        records.append(
            (
                run.name,
                str(config.get("checkpoint", "unknown")),
                str(config.get("data_augmentation", "unknown")),
                dict(run.summary),
            )
        )
    return records


def _print_report(report: AuditReport) -> None:
    """Print the human-readable verdict."""
    print(f"Label distribution: mean {report.label_mean:.2f}, std {report.label_std:.2f}\n")
    print(f"Runs audited: {report.n_runs}")
    print(f"Collapsed to a constant output: {report.n_collapsed} ({100 * report.n_collapsed / max(1, report.n_runs):.1f}%)")
    print("  A collapsed run has std ~0 or an undefined Pearson. metrics._sanitize_predictions")
    print("  rewrites its NaN predictions to 0.0, so it enters the aggregate as a zero-predictor.\n")

    header = f"{'checkpoint':32} {'n':>3} {'pearson':>8} {'rmse':>7} {'rmse_affine':>12} {'recover':>8} {'pred_mean':>10} {'pred_std':>9}"
    print(header)
    print("-" * len(header))
    for checkpoint, stats in report.by_checkpoint().items():
        recovered = 1 - stats["rmse_after_affine"] / stats["rmse"] if stats["rmse"] else 0.0
        print(
            f"{checkpoint[:32]:32} {stats['n']:>3.0f} {stats['pearson']:>8.3f} {stats['rmse']:>7.2f} "
            f"{stats['rmse_after_affine']:>12.2f} {100 * recovered:>7.1f}% {stats['pred_mean']:>10.2f} {stats['pred_std']:>9.2f}"
        )

    print(f"\nFor reference the labels sit at mean {report.label_mean:.2f}, std {report.label_std:.2f}.")
    print("A pred_mean and pred_std well below those, with a healthy Pearson, is amplitude")
    print("compression: the ranking is learned, the scale is not.")


def main() -> None:
    """Audit a wandb sweep for the three H1 calibration failures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default="davebulaval", help="wandb entity.")
    parser.add_argument("--project", default="meaningbert-checkpoint-sweep", help="wandb project.")
    parser.add_argument("--label-mean", type=float, default=DEFAULT_LABEL_MEAN, help="Mean of the gold labels.")
    parser.add_argument("--label-std", type=float, default=DEFAULT_LABEL_STD, help="Std of the gold labels.")
    parser.add_argument("--json-out", default=None, help="Optional path to dump the full per-run audit.")
    args = parser.parse_args()

    records = _fetch_from_wandb(args.entity, args.project)
    report = audit_runs(records, args.label_mean, args.label_std)
    _print_report(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "label_mean": report.label_mean,
                    "label_std": report.label_std,
                    "n_runs": report.n_runs,
                    "n_collapsed": report.n_collapsed,
                    "by_checkpoint": report.by_checkpoint(),
                    "runs": [vars(run) for run in report.runs],
                },
                handle,
                indent=2,
            )
        print(f"\nFull audit written to {args.json_out}")


if __name__ == "__main__":
    main()
