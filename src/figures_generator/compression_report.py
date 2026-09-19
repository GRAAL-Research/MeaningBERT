"""Shared helpers for the result tables: drop the dead runs, show the amplitude compression.

Two corrections of ``docs/H1-diagnostic-calibration.md`` land here.

**C1 downstream.** ``metrics.compute_metrics`` no longer rewrites NaN predictions to 0.0;
it flags the run with a ``diverged`` metric and reports NaN everywhere else. Every table
built from wandb summaries must therefore exclude the flagged runs instead of averaging
their NaN, and must keep excluding the runs of the previous sweep, which carry no flag but
still show the collapse signature (a predicted standard deviation of zero, or a Pearson
that is not a number).

**C5.** A RMSE alone hides the compression: the sweep reported a RMSE of 35 to 54 while
the predictions occupied a quarter of the label amplitude. Reporting ``pred_mean`` and
``pred_std`` next to each RMSE, against the label mean and standard deviation, makes that
visible to anyone reading the table.
"""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any, Iterable, Optional, Sequence

try:  # PYTHONPATH=src, the documented way to run the training and analysis scripts.
    from diagnostics.calibration_audit import COLLAPSE_STD_THRESHOLD, DEFAULT_LABEL_MEAN, DEFAULT_LABEL_STD
except ImportError:  # pragma: no cover - repository root on the path instead of ``src``.
    from src.diagnostics.calibration_audit import (  # type: ignore[no-redef]
        COLLAPSE_STD_THRESHOLD,
        DEFAULT_LABEL_MEAN,
        DEFAULT_LABEL_STD,
    )

#: Last segment of the wandb summary keys carrying the C1 divergence flag. The wandb
#: integration of ``Trainer`` prefixes metric names, so the flag shows up as
#: ``test/diverged``, ``eval/diverged`` or ``train/test/identical_sentences_diverged``.
DIVERGED_SUFFIX = "diverged"

#: Summary keys holding the predicted spread, in the order they are looked up.
PRED_STD_KEYS = ("test/st_dev_score", "test_st_dev_score")

#: Summary keys holding the Pearson correlation, in the order they are looked up.
PEARSON_KEYS = ("test/pearson_corr", "test_pearson_corr")


def _as_float(value: Any) -> float:
    """Coerce a wandb summary value to a float, returning NaN when it is not a number.

    Args:
        value: Raw value read from a wandb run summary. wandb hands back ``"NaN"`` as a
            string for some runs, which is exactly the case that must not be trusted.

    Returns:
        The value as a float, or NaN.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def is_flagged_diverged(summary: dict[str, Any]) -> bool:
    """Whether the run carries the C1 divergence flag on any of its evaluations.

    Args:
        summary: A wandb run summary.

    Returns:
        ``True`` when a ``*diverged`` key is set to a non-zero value.
    """
    for key, value in summary.items():
        if key.replace("_", "/").rsplit("/", maxsplit=1)[-1] == DIVERGED_SUFFIX:
            number = _as_float(value)
            if math.isfinite(number) and number != 0.0:
                return True
    return False


def looks_collapsed(summary: dict[str, Any]) -> bool:
    """Whether the run shows the collapse signature, for runs predating the C1 flag.

    Same rule as ``diagnostics.calibration_audit.audit_run``: a usable run needs a finite,
    non-degenerate predicted spread and a defined correlation. A non-finite standard
    deviation is not "unknown", it is what a dead run reports.

    Args:
        summary: A wandb run summary.

    Returns:
        ``True`` when the predictions are constant or the correlation is undefined.
    """
    pred_std = next((_as_float(summary[key]) for key in PRED_STD_KEYS if key in summary), float("nan"))
    pearson = next((_as_float(summary[key]) for key in PEARSON_KEYS if key in summary), float("nan"))
    if math.isnan(pred_std) and math.isnan(pearson) and not any(k in summary for k in PRED_STD_KEYS + PEARSON_KEYS):
        # Nothing to judge on: this summary simply does not report those metrics.
        return False
    return not math.isfinite(pred_std) or pred_std < COLLAPSE_STD_THRESHOLD or not math.isfinite(pearson)


def is_usable_run(summary: dict[str, Any]) -> bool:
    """Whether a run may enter an aggregate.

    Args:
        summary: A wandb run summary.

    Returns:
        ``False`` when the run is flagged diverged (C1) or shows the collapse signature.
    """
    return not is_flagged_diverged(summary) and not looks_collapsed(summary)


def drop_unusable_runs(summaries: Iterable[dict[str, Any]], label: str = "") -> list[dict[str, Any]]:
    """Filter out the diverged and collapsed runs, reporting how many were dropped.

    Args:
        summaries: The wandb run summaries of one group.
        label: Optional group name, used in the message printed when runs are dropped.

    Returns:
        The summaries that may be averaged.
    """
    all_summaries = list(summaries)
    kept = [summary for summary in all_summaries if is_usable_run(summary)]
    n_dropped = len(all_summaries) - len(kept)
    if n_dropped:
        where = f" in {label}" if label else ""
        print(f"  Excluded {n_dropped} diverged or collapsed run(s){where} from the aggregate (C1).")
    return kept


def compression_ratio(pred_std: float, label_std: float = DEFAULT_LABEL_STD) -> float:
    """By how much the predicted amplitude is compressed against the label amplitude.

    Args:
        pred_std: Standard deviation of the predictions.
        label_std: Standard deviation of the gold labels.

    Returns:
        ``label_std / pred_std``, the factor the diagnostic reports as 2 to 4. NaN when the
        predictions are constant or either value is not usable.
    """
    if not math.isfinite(pred_std) or not math.isfinite(label_std) or pred_std <= 0:
        return float("nan")
    return label_std / pred_std


def format_mean_std(values: Sequence[float], digits: int = 2) -> str:
    """Format a list of per-fold values as ``mean +/- std``.

    Args:
        values: The values to summarize.
        digits: Number of decimals.

    Returns:
        ``"n/a"`` when the list is empty, ``"m"`` for a single value, ``"m +/- s"`` otherwise.
    """
    usable = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    if not usable:
        return "n/a"
    if len(usable) == 1:
        return f"{usable[0]:.{digits}f}"
    return f"{mean(usable):.{digits}f} +/- {stdev(usable):.{digits}f}"


def label_reference_line(
    label_mean: float = DEFAULT_LABEL_MEAN,
    label_std: float = DEFAULT_LABEL_STD,
) -> str:
    """One line stating the label distribution the predicted moments must be read against.

    Args:
        label_mean: Mean of the gold labels.
        label_std: Standard deviation of the gold labels.

    Returns:
        A ready to print sentence.
    """
    return (
        f"Reference: the gold labels sit at mean {label_mean:.2f}, std {label_std:.2f}. "
        f"A Pred mean and Pred std well below those, with a healthy Pearson, is amplitude compression."
    )


def label_reference_caption(
    label_mean: float = DEFAULT_LABEL_MEAN,
    label_std: float = DEFAULT_LABEL_STD,
) -> str:
    """The same reference, escaped for a LaTeX caption.

    Args:
        label_mean: Mean of the gold labels.
        label_std: Standard deviation of the gold labels.

    Returns:
        A LaTeX-safe sentence.
    """
    return (
        f"Pred mean and Pred std are the mean and standard deviation of the predictions; "
        f"the gold labels sit at mean {label_mean:.2f}, std {label_std:.2f}. "
        f"Diverged and collapsed runs are excluded."
    )


def pred_moment_columns(summaries: Sequence[dict[str, Any]]) -> dict[str, Optional[float]]:
    """Extract the predicted mean and spread of a group of runs.

    Args:
        summaries: The wandb run summaries of one group, already filtered.

    Returns:
        A dict with ``pred_mean``, ``pred_std`` and ``compression``, each NaN-free or ``None``.
    """
    means = [_as_float(s.get("test/mean_score")) for s in summaries]
    stds = [_as_float(s.get("test/st_dev_score")) for s in summaries]
    means = [value for value in means if math.isfinite(value)]
    stds = [value for value in stds if math.isfinite(value)]
    pred_mean = mean(means) if means else None
    pred_std = mean(stds) if stds else None
    return {
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "compression": compression_ratio(pred_std) if pred_std is not None else None,
    }
