"""Map every corpus onto the common 0-100 meaning-preservation scale.

This module is the single owner of the ``label`` column. Loaders emit ``label_raw`` in
their native scale and leave ``label`` as NaN; see ``src/data/CONTRACT.md`` for why that
split exists.

Three mapping families, in decreasing order of trustworthiness:

1. :class:`AnchorMap` - fitted on pairs that two corpora genuinely share. This is the only
   family that observes how the two annotation protocols actually relate. Hypothesis H3 of
   ``PRODUIT.md`` bets that the 60 sentences shared between SimpEval2022 and
   SynthSimpliEval are enough of an anchor.
2. :class:`QuantileMap` - matches the source distribution onto the reference distribution.
   It assumes both corpora sample systems of comparable quality, which is usually false.
   Always emits a warning.
3. :class:`BoundsMap` - stretches the declared scale bounds linearly onto 0-100. It assumes
   a Likert 3 means the same thing as a direct-assessment 50, which is not true of any
   annotation protocol ever published. It is the fallback of last resort.

Whichever is used is recorded per corpus in the :class:`HarmonisationReport`, because
hypothesis H2 of ``PRODUIT.md`` can only be tested if the mapping is explicit.
"""

from __future__ import annotations

import math
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from datasets import Dataset, concatenate_datasets

from data.schema import SCALES, ScaleSpec

TARGET_LOW: float = 0.0
TARGET_HIGH: float = 100.0

#: Below this many shared pairs, an anchor fit is noise. Chosen so that the 60 pairs of
#: hypothesis H3 clear the bar with margin while a handful of coincidental matches do not.
MIN_ANCHORS_AFFINE: int = 20
#: Isotonic regression has far more freedom than an affine fit, so it needs more support.
MIN_ANCHORS_ISOTONIC: int = 100

_WHITESPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Collapse whitespace so that two corpora quoting the same sentence match.

    Case and punctuation are preserved: two sentences differing by a capital letter are
    different sentences for a meaning-preservation metric, and silently merging them
    would inflate the anchor count with pairs that are not actually shared.
    """
    return _WHITESPACE.sub(" ", text).strip()


def pair_key(original: str, simplification: str) -> str:
    """Stable identity of a sentence pair, used for anchoring and deduplication."""
    return f"{normalise_text(original)}|||{normalise_text(simplification)}"


# --- Mappings -----------------------------------------------------------------------


class ScaleMap(ABC):
    """Maps ``label_raw`` values of one corpus onto the common 0-100 scale."""

    #: Short identifier recorded in the harmonisation report.
    name: str = "abstract"

    @abstractmethod
    def _apply(self, values: np.ndarray) -> np.ndarray:
        """Map raw values, before clipping."""

    def __call__(self, values: Sequence[float]) -> np.ndarray:
        """Map *values* onto 0-100, clipped to the target range."""
        mapped = self._apply(np.asarray(values, dtype=float))
        return np.clip(mapped, TARGET_LOW, TARGET_HIGH)


@dataclass
class BoundsMap(ScaleMap):
    """Linear stretch of a scale's declared bounds onto 0-100.

    Honours the scale's orientation, so ``severity3`` (where 3 is the worst) comes out
    inverted. Refuses unbounded scales, which have no meaningful upper anchor.
    """

    spec: ScaleSpec
    name: str = "bounds"

    def __post_init__(self) -> None:
        if not math.isfinite(self.spec.high) or not math.isfinite(self.spec.low):
            raise ValueError("BoundsMap needs a bounded scale; use SaturatingCountMap for counts")
        if self.spec.high == self.spec.low:
            raise ValueError("BoundsMap needs a scale with a non-zero range")

    def _apply(self, values: np.ndarray) -> np.ndarray:
        unit = (values - self.spec.low) / (self.spec.high - self.spec.low)
        if not self.spec.higher_is_better:
            unit = 1.0 - unit
        return unit * TARGET_HIGH


@dataclass
class SaturatingCountMap(ScaleMap):
    """Map an unbounded error count onto 0-100, saturating at *cap* errors.

    ``error_count`` has no upper bound, so there is no way to stretch it linearly. This
    map declares an explicit cap: *cap* errors or more means zero meaning preserved. The
    cap is an assumption about the corpus, not a property of it, so it is reported.
    """

    cap: float
    name: str = "saturating_count"

    def __post_init__(self) -> None:
        if self.cap <= 0:
            raise ValueError("cap must be positive")

    def _apply(self, values: np.ndarray) -> np.ndarray:
        return (1.0 - values / self.cap) * TARGET_HIGH


@dataclass
class AffineAnchorMap(ScaleMap):
    """Affine map ``a * raw + b`` least-squares fitted on pairs shared with the reference."""

    slope: float
    intercept: float
    n_anchors: int
    anchor_pearson: float
    name: str = "affine_anchor"

    def _apply(self, values: np.ndarray) -> np.ndarray:
        return self.slope * values + self.intercept


@dataclass
class IsotonicAnchorMap(ScaleMap):
    """Monotone non-parametric map fitted on shared pairs.

    Preferred over :class:`AffineAnchorMap` when the two protocols relate monotonically but
    not linearly, which is the norm between a Likert scale and a continuous one. Needs far
    more anchors to be stable.
    """

    knots_x: np.ndarray
    knots_y: np.ndarray
    n_anchors: int
    anchor_pearson: float
    name: str = "isotonic_anchor"

    def _apply(self, values: np.ndarray) -> np.ndarray:
        return np.interp(values, self.knots_x, self.knots_y)


@dataclass
class QuantileMap(ScaleMap):
    """Match the source distribution onto the reference label distribution.

    Assumes both corpora sample systems of comparable quality. That assumption is usually
    wrong: a corpus built from 2023 LLM outputs and one built from 2018 seq2seq outputs do
    not share a quality distribution. Use only when no anchor exists, and read the warning.
    """

    source_quantiles: np.ndarray
    reference_values: np.ndarray
    name: str = "quantile"

    def _apply(self, values: np.ndarray) -> np.ndarray:
        ranks = np.searchsorted(self.source_quantiles, values, side="left")
        ranks = np.clip(ranks, 0, len(self.reference_values) - 1)
        return self.reference_values[ranks]


# --- Anchors ------------------------------------------------------------------------


@dataclass
class Anchors:
    """Pairs a corpus shares with the reference, with both scores side by side."""

    raw: np.ndarray
    reference: np.ndarray

    def __len__(self) -> int:
        return len(self.raw)

    @property
    def pearson(self) -> float:
        """Correlation between the two protocols on the shared pairs."""
        if len(self) < 3 or np.std(self.raw) == 0 or np.std(self.reference) == 0:
            return float("nan")
        return float(np.corrcoef(self.raw, self.reference)[0, 1])


def find_anchors(source: Dataset, reference: Dataset) -> Anchors:
    """Find the pairs *source* and *reference* both contain.

    When either corpus scores the same pair more than once, its scores are averaged so the
    anchor carries one value per side.

    Args:
        source: Corpus to be mapped, carrying ``label_raw``.
        reference: Reference corpus, already on the 0-100 scale in ``label_raw``.

    Returns:
        The aligned raw and reference scores of every shared pair.
    """
    reference_scores: dict[str, list[float]] = {}
    for original, simplification, value in zip(
        reference["original"], reference["simplification"], reference["label_raw"]
    ):
        reference_scores.setdefault(pair_key(original, simplification), []).append(float(value))

    source_scores: dict[str, list[float]] = {}
    for original, simplification, value in zip(source["original"], source["simplification"], source["label_raw"]):
        key = pair_key(original, simplification)
        if key in reference_scores:
            source_scores.setdefault(key, []).append(float(value))

    keys = sorted(source_scores)
    return Anchors(
        raw=np.array([float(np.mean(source_scores[key])) for key in keys]),
        reference=np.array([float(np.mean(reference_scores[key])) for key in keys]),
    )


def fit_affine(anchors: Anchors) -> AffineAnchorMap:
    """Least-squares fit of ``reference ~ a * raw + b`` on the anchors.

    Raises:
        ValueError: If there are too few anchors, or if the raw scores do not vary, which
            would make the slope unidentifiable.
    """
    if len(anchors) < MIN_ANCHORS_AFFINE:
        raise ValueError(f"need at least {MIN_ANCHORS_AFFINE} anchors for an affine fit, got {len(anchors)}")
    if np.std(anchors.raw) == 0:
        raise ValueError("anchor raw scores are constant; the slope is unidentifiable")
    slope, intercept = np.polyfit(anchors.raw, anchors.reference, deg=1)
    return AffineAnchorMap(
        slope=float(slope),
        intercept=float(intercept),
        n_anchors=len(anchors),
        anchor_pearson=anchors.pearson,
    )


def fit_isotonic(anchors: Anchors) -> IsotonicAnchorMap:
    """Monotone fit of the reference scores against the raw scores.

    Raises:
        ValueError: If there are too few anchors for a non-parametric fit to be stable.
    """
    if len(anchors) < MIN_ANCHORS_ISOTONIC:
        raise ValueError(f"need at least {MIN_ANCHORS_ISOTONIC} anchors for an isotonic fit, got {len(anchors)}")
    from sklearn.isotonic import IsotonicRegression  # local import: sklearn is heavy

    order = np.argsort(anchors.raw)
    x = anchors.raw[order]
    y = IsotonicRegression(out_of_bounds="clip").fit_transform(x, anchors.reference[order])
    return IsotonicAnchorMap(knots_x=x, knots_y=y, n_anchors=len(anchors), anchor_pearson=anchors.pearson)


def fit_quantile(source_values: Sequence[float], reference_values: Sequence[float]) -> QuantileMap:
    """Build a distribution-matching map from *source_values* onto *reference_values*."""
    source_sorted = np.sort(np.asarray(source_values, dtype=float))
    grid = np.linspace(0.0, 100.0, num=min(len(source_sorted), 1000))
    return QuantileMap(
        source_quantiles=np.percentile(source_sorted, grid),
        reference_values=np.percentile(np.asarray(reference_values, dtype=float), grid),
    )


# --- Orchestration ------------------------------------------------------------------


@dataclass
class CorpusMapping:
    """How one corpus was placed on the common scale."""

    corpus: str
    scale: str
    mapping: str
    n_rows: int
    n_anchors: int
    anchor_pearson: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class HarmonisationReport:
    """Per-corpus record of the harmonisation, for hypothesis H2."""

    reference: str
    mappings: list[CorpusMapping] = field(default_factory=list)

    @property
    def weakest(self) -> list[CorpusMapping]:
        """Corpora placed without any anchor, i.e. those H2 is most at risk on.

        The reference corpus is excluded: it has no anchors by construction, since it is
        what everything else is anchored against.
        """
        return [m for m in self.mappings if m.n_anchors == 0 and m.mapping != "identity"]

    def summary(self) -> str:
        """Human-readable table of the mappings."""
        lines = [f"reference corpus: {self.reference}", ""]
        lines.append(f"{'corpus':22} {'scale':14} {'mapping':18} {'rows':>7} {'anchors':>8} {'anchor_r':>9}")
        lines.append("-" * 84)
        for mapping in self.mappings:
            anchor_r = "-" if math.isnan(mapping.anchor_pearson) else f"{mapping.anchor_pearson:.3f}"
            lines.append(
                f"{mapping.corpus[:22]:22} {mapping.scale[:14]:14} {mapping.mapping[:18]:18} "
                f"{mapping.n_rows:>7} {mapping.n_anchors:>8} {anchor_r:>9}"
            )
        for mapping in self.mappings:
            for message in mapping.warnings:
                lines.append(f"  ! {mapping.corpus}: {message}")
        return "\n".join(lines)


def choose_map(
    source: Dataset,
    reference: Dataset,
    count_cap: float,
) -> tuple[ScaleMap, Anchors, list[str]]:
    """Pick the most trustworthy mapping available for *source*.

    Tries anchors first, falls back to the declared bounds, and never silently picks the
    quantile map: distribution matching is only ever chosen explicitly.

    Args:
        source: Corpus to place on the common scale.
        reference: Reference corpus, already on 0-100.
        count_cap: Number of errors at which an ``error_count`` corpus scores zero.

    Returns:
        The mapping, the anchors found, and any warnings to record.
    """
    scale = source["scale"][0]
    anchors = find_anchors(source, reference)
    messages: list[str] = []

    if len(anchors) >= MIN_ANCHORS_ISOTONIC:
        return fit_isotonic(anchors), anchors, messages
    if len(anchors) >= MIN_ANCHORS_AFFINE:
        messages.append(f"affine fit on only {len(anchors)} anchors; monotone non-linearity is not modelled")
        return fit_affine(anchors), anchors, messages

    if len(anchors):
        messages.append(f"{len(anchors)} shared pairs is below the {MIN_ANCHORS_AFFINE} needed to fit; ignored")
    if scale == "error_count":
        messages.append(f"no anchor: saturating count map with an assumed cap of {count_cap} errors")
        return SaturatingCountMap(cap=count_cap), anchors, messages

    messages.append(
        "no anchor: falling back to a linear stretch of the declared bounds, which assumes "
        "the two annotation protocols are linearly comparable. This is the weakest link of "
        "the merge and the first thing to question if H2 fails."
    )
    return BoundsMap(SCALES[scale]), anchors, messages


def harmonize(
    corpora: dict[str, Dataset],
    reference: str = "csmd",
    count_cap: float = 3.0,
    overrides: Optional[dict[str, ScaleMap]] = None,
) -> tuple[Dataset, HarmonisationReport]:
    """Place every corpus on the common 0-100 scale and concatenate them.

    The reference corpus must already be on ``da100``; its ``label`` is copied straight
    from ``label_raw``.

    Args:
        corpora: Loader outputs keyed by corpus name. Must contain *reference*.
        reference: Name of the reference corpus.
        count_cap: Errors at which an ``error_count`` corpus scores zero.
        overrides: Explicit mapping per corpus, bypassing :func:`choose_map`. This is how
            a deliberate :class:`QuantileMap` gets used.

    Returns:
        The merged dataset with ``label`` filled, and the report of how each corpus landed.

    Raises:
        KeyError: If *reference* is absent from *corpora*.
        ValueError: If the reference corpus is not on the ``da100`` scale.
    """
    if reference not in corpora:
        raise KeyError(f"reference corpus '{reference}' missing from {sorted(corpora)}")
    reference_dataset = corpora[reference]
    if reference_dataset["scale"][0] != "da100":
        raise ValueError(f"reference corpus must be on the da100 scale, got '{reference_dataset['scale'][0]}'")

    overrides = overrides or {}
    report = HarmonisationReport(reference=reference)
    pieces: list[Dataset] = []

    for name in [reference] + sorted(set(corpora) - {reference}):
        dataset = corpora[name]
        if name == reference:
            mapped = np.asarray(dataset["label_raw"], dtype=float)
            record = CorpusMapping(name, "da100", "identity", len(dataset), 0, float("nan"))
        else:
            if name in overrides:
                scale_map, anchors, messages = overrides[name], find_anchors(dataset, reference_dataset), []
                if isinstance(scale_map, QuantileMap):
                    messages.append("quantile map: assumes both corpora sample systems of comparable quality")
            else:
                scale_map, anchors, messages = choose_map(dataset, reference_dataset, count_cap)
            mapped = scale_map(dataset["label_raw"])
            record = CorpusMapping(
                corpus=name,
                scale=dataset["scale"][0],
                mapping=scale_map.name,
                n_rows=len(dataset),
                n_anchors=len(anchors),
                anchor_pearson=anchors.pearson,
                warnings=messages,
            )
        report.mappings.append(record)
        pieces.append(dataset.remove_columns(["label"]).add_column("label", mapped.tolist()))

    for mapping in report.mappings:
        for message in mapping.warnings:
            warnings.warn(f"{mapping.corpus}: {message}", stacklevel=2)

    merged = concatenate_datasets([piece.select_columns(pieces[0].column_names) for piece in pieces])
    return merged, report


# --- Deduplication ------------------------------------------------------------------


def deduplicate(dataset: Dataset, priority: Sequence[str]) -> tuple[Dataset, dict[str, int]]:
    """Drop pairs that appear in more than one corpus, keeping the highest-priority one.

    Cross-corpus overlap is real: CSMD, SimpEval and SALSA all draw on ASSET. Keeping both
    copies would let the same sentence pair sit in train and in test with two different
    labels, which is a leak dressed up as extra data.

    Args:
        dataset: Merged, harmonised dataset.
        priority: Corpus names, most trusted first. A corpus absent from this list ranks
            after every listed one.

    Returns:
        The deduplicated dataset, and how many rows each corpus lost.
    """
    rank = {name: index for index, name in enumerate(priority)}
    fallback = len(priority)

    best: dict[str, tuple[int, int]] = {}
    for index, (original, simplification, corpus) in enumerate(
        zip(dataset["original"], dataset["simplification"], dataset["corpus"])
    ):
        key = pair_key(original, simplification)
        candidate = (rank.get(corpus, fallback), index)
        if key not in best or candidate < best[key]:
            best[key] = candidate

    keep = sorted(index for _, index in best.values())
    dropped: dict[str, int] = {}
    kept = set(keep)
    for index, corpus in enumerate(dataset["corpus"]):
        if index not in kept:
            dropped[corpus] = dropped.get(corpus, 0) + 1

    return dataset.select(keep), dropped
