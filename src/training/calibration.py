"""Output scale handling for MeaningBERT: bounded heads (C3) and recalibration (C4).

``docs/H1-diagnostic-calibration.md`` establishes that the sweep learned the ranking and
not the scale. The predictions occupy a quarter of the label amplitude, the Pearson stays
flat at 0.78-0.80, and 40 to 55 % of the reported RMSE is pure calibration loss, that is,
error the best affine rescaling of the same predictions would remove.

This module holds both answers.

**C3, the cause.** The regression head is linear and unbounded, trained with MSE against
0-100 targets, so nothing tells it the scale is bounded. :func:`percent_from_logits`
implements the two bounded alternatives, a sigmoid head times 100 and a unit-scale target
rescaled at evaluation time. Both keep the reported metrics on the 0-100 scale, which is
the condition for comparing anything to the published article.

**C4, the net.** :func:`fit_calibrator` fits an affine or isotonic transformation on the
dev split and applies it to the test split. Fitting on the test split is a leak, so it
raises unless the caller explicitly asks for it. Recalibrating a compressed model restores
the scale, it does not restore the information the model never learned.

The whole module works on numpy arrays and, where it matters for training, on torch
tensors, without importing torch.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:  # PYTHONPATH=src, the documented way to run the training scripts.
    from diagnostics.calibration_audit import DEFAULT_LABEL_MEAN, DEFAULT_LABEL_STD, optimal_affine_rmse
except ImportError:  # pragma: no cover - repository root on the path instead of ``src``.
    from src.diagnostics.calibration_audit import (  # type: ignore[no-redef]
        DEFAULT_LABEL_MEAN,
        DEFAULT_LABEL_STD,
        optimal_affine_rmse,
    )

_log = logging.getLogger(__name__)

#: Bounds of the meaning preservation scale. Everything reported to wandb, to the tables
#: and to the article lives here, whatever the head the model was trained with.
SCORE_MIN: float = 0.0
SCORE_MAX: float = 100.0

#: Output heads selectable from the training CLI.
#:
#: ``linear``
#:     The v1 head: one unbounded linear output, MSE against 0-100 targets. Kept as the
#:     default so an existing command reproduces exactly what it used to do.
#: ``sigmoid``
#:     ``100 * sigmoid(logit)``. Bounded by construction, at every step of training.
#: ``normalized``
#:     Linear output trained against targets divided by 100, multiplied back by 100 and
#:     clipped for evaluation. Bounded only at evaluation time.
#: ``clamped``
#:     Linear output on the unit scale, clamped to ``[0, 1]`` inside the loss. Unlike
#:     ``sigmoid`` it can *reach* the endpoints: ``100 * sigmoid(z)`` needs ``z = 4.6`` to
#:     read 99 and ``z = 6.9`` to read 99.9, and the MSE gradient vanishes long before
#:     that, so the predictions on identical pairs pile up just under the ceiling. Measured
#:     on the v2 runs: identical pairs score 96.36 with a spread of 0.58. When the goal is
#:     for identical pairs to read 100 and unrelated pairs 0, the asymptote is the binding
#:     constraint, and a clamp removes it. The cost is a zero gradient outside the range,
#:     which is harmless because every target lies inside it.
OUTPUT_HEADS: tuple[str, ...] = ("linear", "sigmoid", "normalized", "clamped")
DEFAULT_OUTPUT_HEAD: str = "linear"

#: Recalibration methods.
CALIBRATION_METHODS: tuple[str, ...] = ("affine", "isotonic")

#: Splits a calibrator must never be fitted on. Fitting the transformation on the split it
#: is then scored on is a leak: it reports a number no future input can reproduce.
FORBIDDEN_FIT_SPLITS: frozenset[str] = frozenset({"test", "holdout", "holdout_identical", "holdout_unrelated"})

#: Below this many dev rows, isotonic regression interpolates the dev noise. The CSMD dev
#: split holds 95 rows, well under it, which is why the affine fit is the default.
MIN_ISOTONIC_DEV_SIZE: int = 200

#: Beyond this magnitude the sigmoid is saturated to the float64 limits anyway; clipping
#: there keeps ``exp`` from overflowing on a diverged logit.
_SIGMOID_CLIP: float = 500.0


class CalibrationError(ValueError):
    """The calibrator cannot be fitted on the data it was given."""


class CalibrationLeakError(CalibrationError):
    """The caller tried to fit a calibrator on an evaluation split."""


class IsotonicOverfitWarning(UserWarning):
    """The dev split is too small for isotonic regression to generalize."""


def clip_to_score_range(values: Any) -> Any:
    """Clip values to ``[0, 100]``, for a numpy array or a torch tensor.

    Args:
        values: Scores on the 0-100 scale.

    Returns:
        The same container, clipped to the meaning preservation range.
    """
    if hasattr(values, "clamp"):  # torch.Tensor
        return values.clamp(SCORE_MIN, SCORE_MAX)
    return np.clip(values, SCORE_MIN, SCORE_MAX)


def sigmoid_percent(logits: Any) -> Any:
    """Map unbounded logits to ``[0, 100]`` with a sigmoid, for numpy arrays or torch tensors.

    Args:
        logits: Raw model outputs.

    Returns:
        ``100 * sigmoid(logits)``, bounded by construction and stable for extreme inputs.
    """
    if hasattr(logits, "sigmoid"):  # torch.Tensor
        return logits.sigmoid() * SCORE_MAX
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -_SIGMOID_CLIP, _SIGMOID_CLIP)
    return SCORE_MAX / (1.0 + np.exp(-clipped))


def unit_percent(logits: Any) -> Any:
    """Rescale unit-scale outputs to ``[0, 100]`` and clip them.

    The ``normalized`` head stays linear, so nothing stops it from predicting 1.4. The
    clip is what makes the reported score a meaning preservation score rather than an
    arbitrary real number; it only ever applies at evaluation time.

    Args:
        logits: Raw model outputs, trained against targets in ``[0, 1]``.

    Returns:
        ``100 * logits``, clipped to the meaning preservation range.
    """
    if hasattr(logits, "clamp"):  # torch.Tensor
        return clip_to_score_range(logits * SCORE_MAX)
    return clip_to_score_range(np.asarray(logits, dtype=np.float64) * SCORE_MAX)


def percent_from_logits(logits: Any, head: str = DEFAULT_OUTPUT_HEAD) -> Any:
    """Turn the raw outputs of *head* into scores on the 0-100 scale.

    Args:
        logits: Raw model outputs.
        head: One of :data:`OUTPUT_HEADS`.

    Returns:
        The scores on the 0-100 scale. For ``linear`` the outputs are returned untouched,
        which is exactly the v1 behaviour, unbounded included.

    Raises:
        ValueError: If *head* is not a known head.
    """
    if head == "linear":
        return logits
    if head == "sigmoid":
        return sigmoid_percent(logits)
    if head in ("normalized", "clamped"):
        return unit_percent(logits)
    raise ValueError(f"Unknown output head {head!r}. Expected one of {OUTPUT_HEADS}.")


def targets_for_head(labels: Any, head: str = DEFAULT_OUTPUT_HEAD) -> Any:
    """Convert 0-100 labels to the scale the loss of *head* expects.

    Args:
        labels: Gold labels on the 0-100 scale.
        head: One of :data:`OUTPUT_HEADS`.

    Returns:
        The labels unchanged for ``linear``, divided by 100 for the two bounded heads,
        whose loss is computed on the unit scale.

    Raises:
        ValueError: If *head* is not a known head.
    """
    if head == "linear":
        return labels
    if head in ("sigmoid", "normalized", "clamped"):
        return labels / SCORE_MAX
    raise ValueError(f"Unknown output head {head!r}. Expected one of {OUTPUT_HEADS}.")


def unit_from_logits(logits: Any, head: str = DEFAULT_OUTPUT_HEAD) -> Any:
    """Turn the raw outputs of *head* into the unit scale its loss is computed on.

    Args:
        logits: Raw model outputs.
        head: One of :data:`OUTPUT_HEADS`.

    Returns:
        ``sigmoid(logits)`` for the sigmoid head, the logits themselves for the normalized
        head, and the logits divided by 100 for the linear head.

    Raises:
        ValueError: If *head* is not a known head.
    """
    if head == "linear":
        return logits / SCORE_MAX
    if head == "sigmoid":
        return sigmoid_percent(logits) / SCORE_MAX
    if head == "normalized":
        return logits
    if head == "clamped":
        # Clamping inside the loss is what lets the head reach the endpoints exactly.
        # Outside [0, 1] the gradient is zero, which costs nothing: every target is inside.
        if hasattr(logits, "clamp"):  # torch.Tensor
            return logits.clamp(0.0, 1.0)
        return np.clip(logits, 0.0, 1.0)
    raise ValueError(f"Unknown output head {head!r}. Expected one of {OUTPUT_HEADS}.")


def _as_clean_array(values: Iterable[float], name: str) -> np.ndarray:
    """Convert to a 1-D float array, refusing anything that is not usable.

    Args:
        values: The values to convert.
        name: Name used in the error message.

    Returns:
        A 1-D float64 array.

    Raises:
        CalibrationError: If the array is empty or holds NaN or Inf. A calibrator fitted on
            NaN would silently produce NaN scores, which is the C1 failure again.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise CalibrationError(f"{name} is empty.")
    if not np.all(np.isfinite(array)):
        n_bad = int(np.sum(~np.isfinite(array)))
        raise CalibrationError(f"{name} holds {n_bad} non-finite value(s). A diverged run cannot be recalibrated.")
    return array


@dataclass(frozen=True)
class AffineCalibrator:
    """The affine rescaling ``slope * prediction + intercept``, fitted on the dev split.

    Attributes:
        slope: Multiplicative term, which restores the amplitude.
        intercept: Additive term, which restores the position.
        n_fit: Number of rows the fit used.
        fit_split: Name of the split it was fitted on, kept for the audit trail.
    """

    slope: float
    intercept: float
    n_fit: int
    fit_split: str

    method: str = "affine"

    def transform(self, predictions: Iterable[float], clip: bool = True) -> np.ndarray:
        """Apply the rescaling.

        Args:
            predictions: Predictions on the 0-100 scale.
            clip: Keep the output inside ``[0, 100]``. Only turn it off to check the raw
                fit against the theoretical floor.

        Returns:
            The recalibrated predictions.
        """
        array = np.asarray(predictions, dtype=np.float64)
        rescaled = self.slope * array + self.intercept
        return clip_to_score_range(rescaled) if clip else rescaled

    def describe(self) -> str:
        """A one-line summary, for logs and for the wandb artifact metadata."""
        return (
            f"affine(slope={self.slope:.4f}, intercept={self.intercept:.4f}) "
            f"fitted on {self.n_fit} rows of {self.fit_split!r}"
        )


@dataclass(frozen=True)
class IsotonicCalibrator:
    """A monotone, non-parametric rescaling fitted on the dev split.

    It can correct a non-linear compression an affine fit cannot, at the cost of
    interpolating the dev noise when the dev split is small. See :data:`MIN_ISOTONIC_DEV_SIZE`.

    Attributes:
        model: The fitted ``sklearn.isotonic.IsotonicRegression``.
        n_fit: Number of rows the fit used.
        fit_split: Name of the split it was fitted on.
    """

    model: Any
    n_fit: int
    fit_split: str

    method: str = "isotonic"

    def transform(self, predictions: Iterable[float], clip: bool = True) -> np.ndarray:
        """Apply the rescaling.

        Args:
            predictions: Predictions on the 0-100 scale.
            clip: Keep the output inside ``[0, 100]``. The isotonic fit is already bounded
                by its own ``y_min``/``y_max``, so this is belt and braces.

        Returns:
            The recalibrated predictions.
        """
        array = np.asarray(predictions, dtype=np.float64).ravel()
        rescaled = np.asarray(self.model.predict(array), dtype=np.float64)
        return clip_to_score_range(rescaled) if clip else rescaled

    def describe(self) -> str:
        """A one-line summary, for logs and for the wandb artifact metadata."""
        return f"isotonic fitted on {self.n_fit} rows of {self.fit_split!r}"


def fit_calibrator(
    predictions: Iterable[float],
    labels: Iterable[float],
    *,
    split: str,
    method: str = "affine",
    allow_evaluation_split: bool = False,
) -> AffineCalibrator | IsotonicCalibrator:
    """Fit a recalibration on one split, naming that split out loud.

    *split* is mandatory and has no default on purpose: a caller cannot fit on the test
    split without writing ``split="test"``, and that call raises.

    Args:
        predictions: Model predictions on the 0-100 scale.
        labels: Gold labels on the 0-100 scale.
        split: Name of the split these rows come from, typically ``"dev"``.
        method: One of :data:`CALIBRATION_METHODS`.
        allow_evaluation_split: Escape hatch to fit on an evaluation split anyway. The
            resulting numbers are not reportable; a warning says so.

    Returns:
        The fitted calibrator.

    Raises:
        CalibrationLeakError: If *split* is an evaluation split and the escape hatch is off.
        CalibrationError: If the inputs are unusable: different lengths, non-finite values,
            or constant predictions, which carry no ranking to rescale.
        ValueError: If *method* is unknown.
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method {method!r}. Expected one of {CALIBRATION_METHODS}.")

    normalized_split = split.strip().lower()
    if normalized_split in FORBIDDEN_FIT_SPLITS:
        if not allow_evaluation_split:
            raise CalibrationLeakError(
                f"Refusing to fit a calibrator on split {split!r}: fitting the rescaling on the split it is then "
                f"scored on is a leak. Fit on the dev split and apply to the test split, for instance with "
                f"calibrate_dev_to_test(). Pass allow_evaluation_split=True only for a diagnostic that is never "
                f"reported as a result."
            )
        warnings.warn(
            f"Calibrator fitted on split {split!r}. The resulting scores are leaked and must not be reported.",
            UserWarning,
            stacklevel=2,
        )

    prediction_array = _as_clean_array(predictions, "predictions")
    label_array = _as_clean_array(labels, "labels")
    if prediction_array.size != label_array.size:
        raise CalibrationError(
            f"predictions and labels must have the same length, got {prediction_array.size} and {label_array.size}."
        )

    if method == "affine":
        return _fit_affine(prediction_array, label_array, normalized_split)
    return _fit_isotonic(prediction_array, label_array, normalized_split)


def _fit_affine(predictions: np.ndarray, labels: np.ndarray, split: str) -> AffineCalibrator:
    """Least squares fit of ``labels ~ slope * predictions + intercept``.

    Args:
        predictions: Clean predictions.
        labels: Clean labels.
        split: Name of the split, for the audit trail.

    Returns:
        The fitted calibrator.

    Raises:
        CalibrationError: If the predictions are constant. There is no amplitude to
            restore, and returning a constant predictor would dress a dead model up as a
            calibrated one, which is the failure C1 fixes.
    """
    prediction_variance = float(np.var(predictions))
    if prediction_variance <= 0.0:
        raise CalibrationError(
            "The predictions are constant: there is no ranking to rescale. This is a collapsed run, "
            "not a calibration problem."
        )
    slope = float(np.cov(predictions, labels, ddof=0)[0, 1] / prediction_variance)
    intercept = float(np.mean(labels) - slope * np.mean(predictions))
    return AffineCalibrator(slope=slope, intercept=intercept, n_fit=predictions.size, fit_split=split)


def _fit_isotonic(predictions: np.ndarray, labels: np.ndarray, split: str) -> IsotonicCalibrator:
    """Fit a bounded, monotone rescaling, warning when the split is too small.

    Args:
        predictions: Clean predictions.
        labels: Clean labels.
        split: Name of the split, for the audit trail.

    Returns:
        The fitted calibrator.
    """
    from sklearn.isotonic import IsotonicRegression  # pylint: disable=import-outside-toplevel

    if predictions.size < MIN_ISOTONIC_DEV_SIZE:
        warnings.warn(
            f"Isotonic calibration fitted on {predictions.size} rows, below the {MIN_ISOTONIC_DEV_SIZE} rows needed "
            f"for it to generalize. It will interpolate the noise of this split. Prefer method='affine', which has "
            f"two parameters, or report both and compare them on a split neither was fitted on.",
            IsotonicOverfitWarning,
            stacklevel=3,
        )
    model = IsotonicRegression(y_min=SCORE_MIN, y_max=SCORE_MAX, out_of_bounds="clip", increasing=True)
    model.fit(predictions, labels)
    return IsotonicCalibrator(model=model, n_fit=predictions.size, fit_split=split)


def calibrate_dev_to_test(
    dev_predictions: Iterable[float],
    dev_labels: Iterable[float],
    test_predictions: Iterable[float],
    method: str = "affine",
) -> tuple[np.ndarray, AffineCalibrator | IsotonicCalibrator]:
    """Fit the rescaling on the dev split and apply it to the test split.

    This is the only shape of the operation that is reportable, which is why it is the
    convenient one: the argument names leave no room for fitting on the test predictions.

    Args:
        dev_predictions: Dev predictions on the 0-100 scale.
        dev_labels: Dev gold labels on the 0-100 scale.
        test_predictions: Test predictions on the 0-100 scale.
        method: One of :data:`CALIBRATION_METHODS`.

    Returns:
        The recalibrated test predictions, clipped to ``[0, 100]``, and the calibrator.
    """
    calibrator = fit_calibrator(dev_predictions, dev_labels, split="dev", method=method)
    _log.info("Calibration: %s", calibrator.describe())
    return calibrator.transform(test_predictions), calibrator


def rmse(predictions: Iterable[float], labels: Iterable[float]) -> float:
    """Root mean squared error, on whatever scale the inputs use.

    Args:
        predictions: Predictions.
        labels: Gold labels.

    Returns:
        The RMSE.
    """
    prediction_array = np.asarray(predictions, dtype=np.float64).ravel()
    label_array = np.asarray(labels, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((label_array - prediction_array) ** 2)))


def calibration_gain(
    predictions: Iterable[float],
    labels: Iterable[float],
    calibrated: Iterable[float],
) -> dict[str, float]:
    """Measure what a recalibration recovered, against the floor it cannot beat.

    The floor is ``sigma_y * sqrt(1 - r^2)``, computed by
    ``diagnostics.calibration_audit.optimal_affine_rmse``: no rescaling of these
    predictions can do better, because what is left is the part of the error the
    correlation cannot explain.

    Args:
        predictions: Raw test predictions on the 0-100 scale.
        labels: Test gold labels on the 0-100 scale.
        calibrated: The recalibrated test predictions.

    Returns:
        The RMSE before and after, the theoretical floor, the Pearson correlation, and the
        share of the original RMSE the recalibration removed.
    """
    prediction_array = _as_clean_array(predictions, "predictions")
    label_array = _as_clean_array(labels, "labels")
    pearson = float(np.corrcoef(prediction_array, label_array)[0, 1])
    rmse_before = rmse(prediction_array, label_array)
    rmse_after = rmse(calibrated, label_array)
    floor = optimal_affine_rmse(float(np.std(label_array)), pearson)
    return {
        "pearson": pearson,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "rmse_affine_floor": floor,
        "recovered_fraction": 0.0 if rmse_before <= 0 else max(0.0, (rmse_before - rmse_after) / rmse_before),
    }


def synthetic_predictions(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    n: int,
    pearson: float,
    label_mean: float = DEFAULT_LABEL_MEAN,
    label_std: float = DEFAULT_LABEL_STD,
    pred_mean: float = 31.31,
    pred_std: float = 14.49,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build predictions whose *sample* correlation with the labels is exactly *pearson*.

    Used to demonstrate that the affine recalibration reaches the theoretical floor, and to
    reproduce the compression of the sweep on demand: predictions centred well below the
    labels, with a fraction of their spread.

    Args:
        n: Number of rows.
        pearson: Target sample correlation, in ``[-1, 1]``.
        label_mean: Mean of the generated labels.
        label_std: Population standard deviation of the generated labels.
        pred_mean: Mean of the generated predictions.
        pred_std: Population standard deviation of the generated predictions.
        seed: Seed of the generator.

    Returns:
        The ``(predictions, labels)`` pair.

    Raises:
        ValueError: If *n* is below 3 or *pearson* is outside ``[-1, 1]``.
    """
    if n < 3:
        raise ValueError(f"n must be at least 3, got {n}.")
    if not -1.0 <= pearson <= 1.0:
        raise ValueError(f"pearson must be in [-1, 1], got {pearson}.")

    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    noise = rng.normal(size=n)
    base -= base.mean()
    noise -= noise.mean()
    # Make the noise orthogonal to the signal, so the sample correlation is exactly the
    # requested one rather than the requested one plus a sampling error.
    noise -= (noise @ base) / (base @ base) * base
    base /= np.linalg.norm(base)
    noise /= np.linalg.norm(noise)

    correlated = pearson * base + math.sqrt(max(0.0, 1.0 - pearson**2)) * noise
    labels = label_mean + label_std * base * math.sqrt(n)
    predictions = pred_mean + pred_std * correlated / float(np.std(correlated))
    return predictions, labels


def _demo(n: int, pearson: float, label_std: float, seed: int) -> dict[str, float]:
    """Run the affine recalibration on synthetic data and compare it to the floor.

    Args:
        n: Number of synthetic rows.
        pearson: Correlation between the synthetic predictions and labels.
        label_std: Standard deviation of the synthetic labels.
        seed: Seed of the generator.

    Returns:
        The measurements of :func:`calibration_gain`.
    """
    predictions, labels = synthetic_predictions(n, pearson, label_std=label_std, seed=seed)
    half = n // 2
    calibrated, calibrator = calibrate_dev_to_test(
        predictions[:half], labels[:half], predictions[half:], method="affine"
    )
    gain = calibration_gain(predictions[half:], labels[half:], calibrated)
    print(f"Synthetic sample: n={n}, pearson={pearson:.3f}, label std={label_std:.2f}")
    print(f"Calibrator: {calibrator.describe()}")
    print(f"RMSE reported        : {gain['rmse_before']:.2f}")
    print(f"RMSE after affine    : {gain['rmse_after']:.2f}")
    print(f"Affine floor         : {gain['rmse_affine_floor']:.2f}   (sigma_y * sqrt(1 - r^2))")
    print(f"Share recovered      : {100 * gain['recovered_fraction']:.1f} %")
    return gain


def main() -> None:
    """Demonstrate on synthetic data that the affine fit reaches the theoretical floor."""
    import argparse  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--n", type=int, default=2000, help="Number of synthetic rows.")
    parser.add_argument("--pearson", type=float, default=0.80, help="Correlation of the synthetic predictions.")
    parser.add_argument("--label-std", type=float, default=DEFAULT_LABEL_STD, help="Std of the synthetic labels.")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the generator.")
    args = parser.parse_args()
    _demo(args.n, args.pearson, args.label_std, args.seed)


if __name__ == "__main__":
    main()
