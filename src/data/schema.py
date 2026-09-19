"""Normalised schema shared by every CSMD v2 corpus loader.

See ``src/data/CONTRACT.md`` for the rationale. The short version: a loader emits
``label_raw`` in its native scale and leaves ``label`` as NaN. Only
``src/data/harmonize.py`` is allowed to fill ``label``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from datasets import Dataset

# --- Vocabularies -------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleSpec:
    """Bounds and orientation of a native annotation scale."""

    low: float
    high: float
    higher_is_better: bool


SCALES: Final[dict[str, ScaleSpec]] = {
    "da100": ScaleSpec(0.0, 100.0, True),
    "likert5": ScaleSpec(1.0, 5.0, True),
    "likert7": ScaleSpec(1.0, 7.0, True),
    "severity3": ScaleSpec(1.0, 3.0, False),
    # Signed 3-point Likert, as used by the TREC PLABA expert judgements: -1 wrong,
    # 0 partial, 1 correct. Distinct from severity3 despite the same cardinality: the
    # bounds differ and the orientation is the opposite, so reusing severity3 would
    # silently invert the signal.
    "likert3_signed": ScaleSpec(-1.0, 1.0, True),
    "binary": ScaleSpec(0.0, 1.0, True),
    "error_count": ScaleSpec(0.0, math.inf, False),
}

SOURCES: Final[frozenset[str]] = frozenset({"original", "identical", "unrelated", "back_translated"})
DOMAINS: Final[frozenset[str]] = frozenset({"wiki", "news", "biomedical", "scientific", "mixed"})
SPLIT_HINTS: Final[frozenset[str]] = frozenset({"train", "dev", "test", ""})

STR_COLUMNS: Final[tuple[str, ...]] = (
    "item_id",
    "original",
    "simplification",
    "scale",
    "corpus",
    "source",
    "domain",
    "system",
    "split_hint",
    "license",
)
FLOAT_COLUMNS: Final[tuple[str, ...]] = ("label_raw", "label", "label_std")
INT_COLUMNS: Final[tuple[str, ...]] = ("n_annotators",)

COLUMNS: Final[tuple[str, ...]] = STR_COLUMNS + FLOAT_COLUMNS + INT_COLUMNS


class ContractError(ValueError):
    """Raised when a dataset violates the CSMD v2 loader contract."""


# --- Construction helper ------------------------------------------------------------


def build(rows: list[dict], corpus: str) -> Dataset:
    """Build a contract-compliant :class:`Dataset` from loosely typed rows.

    Fills the columns a loader should never have to think about (``label`` is always NaN,
    ``corpus`` is constant) and applies the defaults documented in the contract.

    Args:
        rows: One dict per pair. Must carry at least ``item_id``, ``original``,
            ``simplification``, ``label_raw``, ``scale``, ``domain``, ``license``.
        corpus: Corpus identifier, also used as the ``item_id`` prefix.

    Returns:
        A dataset with exactly the contract columns, in contract order.

    Raises:
        ContractError: If a required key is missing from a row.
    """
    required = {"item_id", "original", "simplification", "label_raw", "scale", "domain", "license"}
    built: dict[str, list] = {column: [] for column in COLUMNS}

    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ContractError(f"row {index}: missing required keys {sorted(missing)}")

        item_id = str(row["item_id"])
        built["item_id"].append(item_id if item_id.startswith(f"{corpus}:") else f"{corpus}:{item_id}")
        built["original"].append(str(row["original"]).strip())
        built["simplification"].append(str(row["simplification"]).strip())
        built["label_raw"].append(float(row["label_raw"]))
        built["label"].append(float("nan"))
        built["scale"].append(str(row["scale"]))
        built["n_annotators"].append(int(row.get("n_annotators", 0)))
        built["label_std"].append(float(row.get("label_std", float("nan"))))
        built["corpus"].append(corpus)
        built["source"].append(str(row.get("source", "original")))
        built["domain"].append(str(row["domain"]))
        built["system"].append(str(row.get("system", "")))
        built["split_hint"].append(str(row.get("split_hint", "")))
        built["license"].append(str(row["license"]))

    return Dataset.from_dict(built)


# --- Validation ---------------------------------------------------------------------


def _check_columns(dataset: Dataset, problems: list[str]) -> bool:
    """Append column-level problems. Returns False when the shape is too broken to go on."""
    actual = set(dataset.column_names)
    expected = set(COLUMNS)
    if actual != expected:
        if expected - actual:
            problems.append(f"missing columns: {sorted(expected - actual)}")
        if actual - expected:
            problems.append(f"unexpected columns: {sorted(actual - expected)}")
        return False
    return True


def _check_scales(dataset: Dataset, problems: list[str]) -> None:
    """Append problems about the ``scale`` column and ``label_raw`` bounds."""
    scales = set(dataset["scale"])
    unknown = scales - SCALES.keys()
    if unknown:
        problems.append(f"unknown scale(s) {sorted(unknown)}; permitted: {sorted(SCALES)}")
        return
    if len(scales) > 1:
        problems.append(f"a loader must emit a single scale, found {sorted(scales)}")
        return

    spec = SCALES[next(iter(scales))]
    out_of_bounds = [
        value for value in dataset["label_raw"] if not math.isfinite(value) or not spec.low <= value <= spec.high
    ]
    if out_of_bounds:
        preview = out_of_bounds[:5]
        problems.append(
            f"{len(out_of_bounds)} label_raw value(s) outside [{spec.low}, {spec.high}] or non-finite, e.g. {preview}"
        )


def _check_vocabularies(dataset: Dataset, problems: list[str]) -> None:
    """Append problems about the closed-vocabulary string columns."""
    for column, vocabulary in (("source", SOURCES), ("domain", DOMAINS), ("split_hint", SPLIT_HINTS)):
        unknown = set(dataset[column]) - vocabulary
        if unknown:
            problems.append(f"{column}: unknown value(s) {sorted(unknown)}; permitted: {sorted(vocabulary)}")


def _check_identity(dataset: Dataset, problems: list[str]) -> None:
    """Append problems about ``corpus``, ``item_id`` uniqueness and text emptiness."""
    corpora = set(dataset["corpus"])
    if len(corpora) != 1:
        problems.append(f"a loader must emit a single corpus, found {sorted(corpora)}")
        return
    corpus = next(iter(corpora))

    item_ids = dataset["item_id"]
    if len(set(item_ids)) != len(item_ids):
        problems.append(f"item_id is not unique ({len(item_ids) - len(set(item_ids))} duplicate(s))")
    bad_prefix = [item_id for item_id in item_ids if not item_id.startswith(f"{corpus}:")]
    if bad_prefix:
        problems.append(f"{len(bad_prefix)} item_id(s) not prefixed by '{corpus}:', e.g. {bad_prefix[:3]}")

    for column in ("original", "simplification"):
        empty = sum(1 for text in dataset[column] if not text.strip())
        if empty:
            problems.append(f"{column}: {empty} empty value(s)")


def _check_labels(dataset: Dataset, problems: list[str]) -> None:
    """Append problems about ``label``, ``n_annotators`` and ``label_std``."""
    filled = sum(1 for value in dataset["label"] if not math.isnan(value))
    if filled:
        problems.append(f"label must be NaN in a loader ({filled} value(s) filled); harmonize.py owns that column")

    negative = sum(1 for value in dataset["n_annotators"] if value < 0)
    if negative:
        problems.append(f"n_annotators: {negative} negative value(s)")

    bad_std = sum(
        1
        for count, std in zip(dataset["n_annotators"], dataset["label_std"])
        if count >= 2 and math.isfinite(std) and std < 0
    )
    if bad_std:
        problems.append(f"label_std: {bad_std} negative value(s)")


def validate(dataset: Dataset) -> None:
    """Raise :class:`ContractError` listing every way *dataset* breaks the loader contract.

    Args:
        dataset: Output of a loader's ``load()``.

    Raises:
        ContractError: If the dataset is empty or violates any contract rule. The message
            lists every problem found, not just the first.
    """
    problems: list[str] = []

    if len(dataset) == 0:
        raise ContractError("dataset is empty")

    if not _check_columns(dataset, problems):
        raise ContractError("contract violated:\n  - " + "\n  - ".join(problems))

    _check_identity(dataset, problems)
    _check_scales(dataset, problems)
    _check_vocabularies(dataset, problems)
    _check_labels(dataset, problems)

    if problems:
        raise ContractError("contract violated:\n  - " + "\n  - ".join(problems))
