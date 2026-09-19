"""Training callbacks for MeaningBERT.

Correction C2 of ``docs/H1-diagnostic-calibration.md``. One run of the sweep collapsed to
a constant output at epoch 2 (``pred_std = 0.00``), diverged to NaN at epoch 44, and kept
burning GPU until epoch 58. Nothing stopped it and nothing said so. This module watches
the spread of the evaluation predictions and stops the run once it has been degenerate for
a configurable number of consecutive evaluations.

The decision logic lives in :class:`CollapseDetector`, which depends on nothing but the
standard library, so it is testable without torch or transformers.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

try:  # transformers is installed on the training server, not necessarily on a dev machine.
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:  # pragma: no cover - exercised on machines without transformers.
    _TrainerCallback = object  # type: ignore[assignment, misc]

try:  # PYTHONPATH=src, the documented way to run the training scripts.
    from diagnostics.calibration_audit import COLLAPSE_STD_THRESHOLD
except ImportError:  # pragma: no cover - repository root on the path instead of ``src``.
    from src.diagnostics.calibration_audit import COLLAPSE_STD_THRESHOLD  # type: ignore[no-redef]

_log = logging.getLogger(__name__)

#: Number of consecutive degenerate evaluations before a run is declared collapsed.
#: Three evaluations, so a single unlucky epoch never stops a healthy run.
DEFAULT_COLLAPSE_PATIENCE: int = 3

#: Metric names the callback reads, as ``Trainer`` prefixes them during training.
DEFAULT_STD_METRIC_NAME: str = "eval_st_dev_score"
DEFAULT_DIVERGED_METRIC_NAME: str = "eval_diverged"


class CollapseDetector:
    """Decide whether a run has collapsed to a constant output.

    A run is degenerate at a given evaluation when the standard deviation of its
    predictions is below *std_threshold*, or is not a finite number at all. A non-finite
    spread is not "unknown", it is what a dead run reports.

    Args:
        std_threshold: Spread below which the predictions count as constant. Defaults to
            the threshold the H1 audit uses to call a run collapsed.
        patience: Number of *consecutive* degenerate evaluations before the verdict is
            collapsed. Must be at least 1.

    Raises:
        ValueError: If *patience* is below 1 or *std_threshold* is negative.
    """

    def __init__(
        self,
        std_threshold: float = COLLAPSE_STD_THRESHOLD,
        patience: int = DEFAULT_COLLAPSE_PATIENCE,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}.")
        if std_threshold < 0 or not math.isfinite(std_threshold):
            raise ValueError(f"std_threshold must be a finite non-negative number, got {std_threshold}.")
        self.std_threshold = std_threshold
        self.patience = patience
        self._consecutive = 0

    @property
    def consecutive_degenerate(self) -> int:
        """How many consecutive degenerate evaluations have been observed."""
        return self._consecutive

    @property
    def collapsed(self) -> bool:
        """Whether the run has been degenerate for *patience* consecutive evaluations."""
        return self._consecutive >= self.patience

    def reset(self) -> None:
        """Forget the observations made so far."""
        self._consecutive = 0

    def update(self, pred_std: Optional[float]) -> bool:
        """Record one evaluation and return the verdict.

        Args:
            pred_std: Standard deviation of the evaluation predictions, or ``None`` when
                the metric is missing. A missing metric is not evidence either way: it
                leaves the counter untouched.

        Returns:
            ``True`` once *patience* consecutive degenerate evaluations have been seen.
        """
        if pred_std is None:
            _log.warning("CollapseDetector: no prediction spread reported for this evaluation, verdict unchanged.")
            return self.collapsed

        if self.is_degenerate(pred_std):
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self.collapsed

    def is_degenerate(self, pred_std: float) -> bool:
        """Whether a single evaluation shows a degenerate spread.

        Args:
            pred_std: Standard deviation of the evaluation predictions.

        Returns:
            ``True`` when the spread is non-finite or below the threshold.
        """
        return not math.isfinite(pred_std) or pred_std < self.std_threshold

    def reason(self) -> str:
        """A one-line explanation of the current verdict, for logs and wandb."""
        return (
            f"predictions had a standard deviation below {self.std_threshold} "
            f"(or not a number) for {self._consecutive} consecutive evaluations"
        )


class PredictionCollapseCallback(_TrainerCallback):  # type: ignore[misc, valid-type]
    """Stop a training run whose evaluation predictions have collapsed to a constant.

    Args:
        std_threshold: Spread below which the predictions count as constant.
        patience: Number of consecutive degenerate evaluations before stopping.
        stop_on_diverged: Stop immediately when ``compute_metrics`` flags the evaluation
            as diverged (C1). NaN predictions are not recoverable, so there is nothing to
            gain from waiting for *patience* evaluations.
        std_metric_name: Name of the spread metric in the dict ``Trainer`` hands over.
        diverged_metric_name: Name of the C1 divergence flag in the same dict.
    """

    def __init__(
        self,
        std_threshold: float = COLLAPSE_STD_THRESHOLD,
        patience: int = DEFAULT_COLLAPSE_PATIENCE,
        stop_on_diverged: bool = True,
        std_metric_name: str = DEFAULT_STD_METRIC_NAME,
        diverged_metric_name: str = DEFAULT_DIVERGED_METRIC_NAME,
    ) -> None:
        self.detector = CollapseDetector(std_threshold=std_threshold, patience=patience)
        self.stop_on_diverged = stop_on_diverged
        self.std_metric_name = std_metric_name
        self.diverged_metric_name = diverged_metric_name
        #: Set when the callback stopped the run, so the caller can report why.
        self.stop_reason: Optional[str] = None

    def _lookup(self, metrics: dict[str, Any], name: str) -> Optional[float]:
        """Read *name* from *metrics*, falling back to any key with the same suffix.

        ``Trainer`` prefixes metric names with the evaluation prefix, which differs between
        the training loop and the explicit test evaluations.

        Args:
            metrics: The metric dict handed over by ``Trainer``.
            name: The expected metric name.

        Returns:
            The value as a float, or ``None`` when no matching key holds a number.
        """
        suffix = name.split("_", maxsplit=1)[-1]
        candidates = [name] + [key for key in metrics if key.endswith(suffix)]
        for key in candidates:
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    return float("nan")
        return None

    def on_evaluate(  # pylint: disable=unused-argument
        self, args: Any, state: Any, control: Any, metrics: Optional[dict] = None, **kwargs: Any
    ) -> Any:
        """Check the evaluation spread and stop the run when it has collapsed.

        Args:
            args: ``TrainingArguments``, unused.
            state: ``TrainerState``, used only for the epoch number in the log message.
            control: ``TrainerControl``, whose ``should_training_stop`` we set.
            metrics: The metrics just computed.
            **kwargs: Ignored extras passed by ``Trainer``.

        Returns:
            The (possibly modified) *control*.
        """
        if not metrics:
            return control

        epoch = getattr(state, "epoch", None)
        if self.stop_on_diverged:
            diverged = self._lookup(metrics, self.diverged_metric_name)
            if diverged is not None and math.isfinite(diverged) and diverged != 0.0:
                self.stop_reason = "predictions contain NaN or Inf (C1 divergence flag)"
                _log.error("Stopping at epoch %s: %s.", epoch, self.stop_reason)
                control.should_training_stop = True
                return control

        if self.detector.update(self._lookup(metrics, self.std_metric_name)):
            self.stop_reason = self.detector.reason()
            _log.error("Stopping at epoch %s: %s.", epoch, self.stop_reason)
            control.should_training_stop = True
        return control
