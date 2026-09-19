"""Metrics computed during training and evaluation, on the 0-100 meaning preservation scale.

A run whose predictions contain NaN or Inf is **flagged, not repaired**. Rewriting those
values to 0.0 turns a dead run into a credible zero-predictor: on a 0-100 scale, 0 is a
legitimate and frequent label (the unrelated pairs), so the resulting RMSE looks plausible
and the run enters the sweep aggregate unnoticed. See ``docs/H1-diagnostic-calibration.md``,
correction C1.
"""

import logging
from typing import Any

import numpy as np
from evaluate import load
from sklearn.metrics import root_mean_squared_error

_log = logging.getLogger(__name__)

r2_metric = load("r_squared")
pearsonr_metric = load("pearsonr")

#: Metric key set to 1.0 when the predictions of a run are not usable, 0.0 otherwise.
#: Downstream aggregation must exclude the runs flagged by this key.
DIVERGED_KEY = "diverged"

#: Metric key carrying how many predictions were not finite.
N_NON_FINITE_KEY = "n_non_finite_predictions"


def _sanitize_predictions(predictions: Any) -> tuple[np.ndarray, int]:
    """Squeeze ``(N, 1) -> (N,)`` and count the non-finite predictions, without rewriting them.

    Args:
        predictions: Raw model outputs, of shape ``(N,)`` or ``(N, 1)``.

    Returns:
        The squeezed predictions and the number of NaN or Inf values they contain. The
        values themselves are left untouched: a caller that gets a non-zero count must
        flag the run, not patch the numbers.
    """
    predictions = np.asarray(predictions, dtype=np.float64).squeeze()
    predictions = np.atleast_1d(predictions)
    n_non_finite = int(np.sum(~np.isfinite(predictions)))
    if n_non_finite > 0:
        _log.error(
            "_sanitize_predictions: %d/%d predictions are NaN or Inf. The run is flagged as diverged; "
            "its metrics are reported as NaN and must be excluded from any aggregate.",
            n_non_finite,
            predictions.size,
        )
    return predictions, n_non_finite


def _diverged_metrics(metric_keys: tuple[str, ...], n_non_finite: int) -> dict[str, float]:
    """Build the metric dict of a diverged run: every number is NaN, the marker is set.

    Args:
        metric_keys: Names of the metrics the caller normally returns.
        n_non_finite: Number of non-finite predictions that triggered the flag.

    Returns:
        A dict with the same keys as a healthy run, all NaN, plus the divergence markers.
    """
    metrics: dict[str, float] = {key: float("nan") for key in metric_keys}
    metrics[DIVERGED_KEY] = 1.0
    metrics[N_NON_FINITE_KEY] = float(n_non_finite)
    return metrics


_TRAIN_METRIC_KEYS = ("rmse", "R2", "pearson_corr", "pearson_pvalue", "mean_score", "st_dev_score")
_IDENTICAL_METRIC_KEYS = ("rmse", "mean_score", "st_dev_score", "ratio_equals", "ratio_95", "ratio_99")
_UNRELATED_METRIC_KEYS = ("rmse", "mean_score", "st_dev_score", "ratio_equals", "ratio_1", "ratio_5")


def compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, Any]:
    """Compute the regression metrics reported during training and on the test set.

    Args:
        eval_pred: The ``(predictions, labels)`` pair handed over by the ``Trainer``.

    Returns:
        The metrics on the 0-100 scale, plus ``diverged`` (0.0 or 1.0). When ``diverged``
        is 1.0 every other value is NaN and the run must be excluded from aggregates.
    """
    predictions, labels = eval_pred
    predictions, n_non_finite = _sanitize_predictions(predictions)
    if n_non_finite > 0:
        return _diverged_metrics(_TRAIN_METRIC_KEYS, n_non_finite)

    rmse = root_mean_squared_error(labels, predictions)
    r_squared = r2_metric.compute(predictions=predictions, references=labels)
    pearson_corr = pearsonr_metric.compute(predictions=predictions, references=labels, return_pvalue=True)
    mean_score = predictions.mean()
    st_dev_score = predictions.std()
    return {
        "rmse": rmse,
        "R2": r_squared,
        "pearson_corr": pearson_corr["pearsonr"],
        "pearson_pvalue": pearson_corr["p-value"],
        "mean_score": mean_score,
        "st_dev_score": st_dev_score,
        DIVERGED_KEY: 0.0,
    }


def eval_compute_metrics_identical(eval_pred: tuple[Any, Any]) -> dict[str, Any]:
    """Compute the sanity-check metrics on the identical sentence holdout.

    We do not compute the correlation and R2 since the labels are all the same, it does not does compute properly.
    E.g. for the R2 the SST score equal 0 since the mean of all labels is 100 and the references are all 100. Thus,
    SSR / 0 is undefined. And 1 - SSR / 1 would be strange.
    See here for compute details https://huggingface.co/spaces/evaluate-metric/r_squared/edit/main/r_squared.py.

    Args:
        eval_pred: The ``(predictions, labels)`` pair handed over by the ``Trainer``.

    Returns:
        The metrics on the 0-100 scale, plus ``diverged`` (0.0 or 1.0).
    """
    predictions, labels = eval_pred
    predictions, n_non_finite = _sanitize_predictions(predictions)
    if n_non_finite > 0:
        return _diverged_metrics(_IDENTICAL_METRIC_KEYS, n_non_finite)

    rmse = root_mean_squared_error(labels, predictions)
    mean_score = predictions.mean()
    st_dev_score = predictions.std()

    # This is only for the hold out test
    counts_95 = [s.round() > 95 for s in predictions]
    ratio_95 = np.multiply(np.divide(sum(counts_95), len(counts_95)), 100).item()
    counts_99 = [s.round() > 99 for s in predictions]
    ratio_99 = np.multiply(np.divide(sum(counts_99), len(counts_99)), 100).item()
    counts_equals = [s.round() == 100 for s in predictions]
    ratio_equals = np.multiply(np.divide(sum(counts_equals), len(counts_equals)), 100).item()

    return {
        "rmse": rmse,
        "mean_score": mean_score,
        "st_dev_score": st_dev_score,
        "ratio_equals": ratio_equals,
        "ratio_95": ratio_95,
        "ratio_99": ratio_99,
        DIVERGED_KEY: 0.0,
    }


def eval_compute_metrics_unrelated(eval_pred: tuple[Any, Any]) -> dict[str, Any]:
    """Compute the sanity-check metrics on the unrelated sentence holdout.

    We do not compute the correlation and R2 since the labels are all the same, it does not does compute properly.
    E.g. for the R2 the SST score equal 0 since the mean of all labels is 100 and the references are all 100. Thus,
    SSR / 0 is undefined. And 1 - SSR / 1 would be strange.
    See here for compute details https://huggingface.co/spaces/evaluate-metric/r_squared/edit/main/r_squared.py.

    Args:
        eval_pred: The ``(predictions, labels)`` pair handed over by the ``Trainer``.

    Returns:
        The metrics on the 0-100 scale, plus ``diverged`` (0.0 or 1.0).
    """
    predictions, labels = eval_pred
    predictions, n_non_finite = _sanitize_predictions(predictions)
    if n_non_finite > 0:
        return _diverged_metrics(_UNRELATED_METRIC_KEYS, n_non_finite)

    rmse = root_mean_squared_error(labels, predictions)
    mean_score = predictions.mean()
    st_dev_score = predictions.std()

    # This is only for the hold out test
    counts_1 = [s.round() < 1 for s in predictions]
    ratio_1 = np.multiply(np.divide(sum(counts_1), len(counts_1)), 100).item()
    counts_5 = [s.round() < 5 for s in predictions]
    ratio_5 = np.multiply(np.divide(sum(counts_5), len(counts_5)), 100).item()
    counts_equals = [s.round() == 0 for s in predictions]
    ratio_equals = np.multiply(np.divide(sum(counts_equals), len(counts_equals)), 100).item()

    return {
        "rmse": rmse,
        "mean_score": mean_score,
        "st_dev_score": st_dev_score,
        "ratio_equals": ratio_equals,
        "ratio_1": ratio_1,
        "ratio_5": ratio_5,
        DIVERGED_KEY: 0.0,
    }
