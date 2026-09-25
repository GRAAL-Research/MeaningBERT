"""Normalised schema shared by every CSMD corpus loader, v2 and v3.

See ``src/data/CONTRACT.md`` for the rationale. The short version: a loader emits
``label_raw`` in its native scale and leaves ``label`` as NaN. Only
``src/data/harmonize.py`` is allowed to fill ``label``.

v3 carries a second target on the same rows. The score became signed, and a signed score
is two questions and not one: how much meaning the pair shares, and whether it is asserted
or denied. So the schema gained ``polarity_raw`` beside ``label_raw``, under the same
discipline, and a corpus is now allowed to answer only one of the two. VitaminC has no
preservation annotation and CSMD has no polarity annotation; both are valid.
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
    # v3. A corpus with no meaning-preservation annotation at all, only polarity: VitaminC,
    # PAWS, MoNLI, NaN-NLI, ACES. Its ``label_raw`` must be NaN on every row. Declaring the
    # absence rather than defaulting to 0.0 is the point: a silent zero would read as "no
    # meaning preserved", which is the opposite of "not measured".
    "none": ScaleSpec(math.nan, math.nan, True),
}

#: Native polarity label spaces, one per family of corpora, each with its CLOSED set of
#: permitted raw values.
#:
#: Why the raw value is a name and not the native integer. ``yangwang825/sick`` encodes its
#: classes as 0, 1 and 2, and which integer means contradiction is not written anywhere in
#: the dataset card. Getting it backwards is silent: the pipeline runs, the model trains,
#: and the only symptom is a number that looks merely disappointing. It already happened
#: once on this project, in ``src/diagnostics/dissociation.py``, where the swap surfaced as
#: an AUC of 0.021 and was caught only because a near-perfectly inverted ranking is too
#: strange to ignore. Forcing the loader to write ``"contradiction"`` puts that decision in
#: the one place that can justify it, next to a report that records the evidence.
#:
#: What the loader does NOT decide is how a scheme maps onto the unified three classes:
#: that belongs to ``harmonize.py``, exactly as the 0-100 rescaling does. The split is
#: between naming a native class, which only the loader can do, and reconciling schemes,
#: which needs the whole picture.
POLARITY_SCHEMES: Final[dict[str, frozenset[str]]] = {
    # A corpus that annotates meaning preservation but never polarity: all four v2 corpora.
    "none": frozenset({""}),
    # The NLI convention: SICK, MoNLI, NaN-NLI. MoNLI carries no contradiction class, which
    # is a property of that corpus and not of the scheme.
    "nli3": frozenset({"entailment", "neutral", "contradiction"}),
    # Fact verification: VitaminC. A claim supported or refuted BY its evidence, which is
    # the same relation under another name.
    "fact3": frozenset({"SUPPORTS", "NOT ENOUGH INFO", "REFUTES"}),
    # Paraphrase identification: PAWS. Deliberately NOT mapped onto nli3 by the loader.
    # "not a paraphrase" is not "a contradiction", and conflating the two is the precise
    # error this corpus exists to detect.
    "paraphrase2": frozenset({"paraphrase", "not_paraphrase"}),
    # Translation adequacy: ACES, whose rows carry a good and an incorrect translation of
    # the same source rather than a class.
    "mt_pair2": frozenset({"good", "incorrect"}),
}

#: The unified three classes the ``polarity`` column holds, after ``harmonize.py`` has
#: reconciled the native schemes. The index is the training target of the polarity head.
#: ``float("nan")`` in that column means "this pair carries no polarity annotation", which
#: is a first-class state and not a missing value: a corpus is allowed to annotate only one
#: of the two targets, and so is a generated row whose polarity cannot be derived.
POLARITY_CLASSES: Final[dict[str, int]] = {"entailment": 0, "neutral": 1, "contradiction": 2}

#: Polarity relations that survive swapping the two sentences. Contradiction is symmetric:
#: if A denies B then B denies A. Entailment is NOT: "a dog is running" entails "an animal
#: is running", and the reverse does not hold. Neutral is not symmetric either, since a
#: pair that is neutral one way round can be an entailment the other way. Carrying a
#: polarity label through a swap is therefore only sound for contradiction.
SYMMETRIC_POLARITIES: Final[frozenset[str]] = frozenset({"contradiction"})

SOURCES: Final[frozenset[str]] = frozenset({"original", "identical", "unrelated", "swapped", "back_translated"})
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
    "polarity_raw",
    "polarity_scheme",
)
FLOAT_COLUMNS: Final[tuple[str, ...]] = ("label_raw", "label", "label_std", "polarity")
INT_COLUMNS: Final[tuple[str, ...]] = ("n_annotators",)

COLUMNS: Final[tuple[str, ...]] = STR_COLUMNS + FLOAT_COLUMNS + INT_COLUMNS


class ContractError(ValueError):
    """Raised when a dataset violates the CSMD v2 loader contract."""


# --- Construction helper ------------------------------------------------------------


def build(rows: list[dict], corpus: str) -> Dataset:
    """Build a contract-compliant :class:`Dataset` from loosely typed rows.

    Fills the columns a loader should never have to think about (``label`` and ``polarity``
    are always NaN, ``corpus`` is constant) and applies the defaults documented in the
    contract. The polarity defaults are what keep the v2 loaders valid untouched: a row
    that says nothing about polarity gets scheme ``none``.

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
        built["polarity_raw"].append(str(row.get("polarity_raw", "")))
        built["polarity_scheme"].append(str(row.get("polarity_scheme", "none")))
        built["polarity"].append(float("nan"))

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

    scale = next(iter(scales))
    if scale == "none":
        measured = sum(1 for value in dataset["label_raw"] if not math.isnan(value))
        if measured:
            problems.append(
                f"scale 'none' declares no meaning-preservation annotation, but {measured} "
                "label_raw value(s) are not NaN"
            )
        return

    spec = SCALES[scale]
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


def _check_polarity(dataset: Dataset, problems: list[str]) -> None:
    """Append problems about ``polarity_scheme``, ``polarity_raw`` and ``polarity``."""
    schemes = set(dataset["polarity_scheme"])
    unknown = schemes - POLARITY_SCHEMES.keys()
    if unknown:
        problems.append(f"unknown polarity_scheme(s) {sorted(unknown)}; permitted: {sorted(POLARITY_SCHEMES)}")
        return
    if len(schemes) > 1:
        problems.append(f"a loader must emit a single polarity_scheme, found {sorted(schemes)}")
        return

    scheme = next(iter(schemes))
    permitted = POLARITY_SCHEMES[scheme]
    off_vocabulary = sorted(set(dataset["polarity_raw"]) - permitted)
    if off_vocabulary:
        problems.append(
            f"polarity_raw: value(s) {off_vocabulary} outside scheme '{scheme}'; permitted: {sorted(permitted)}"
        )

    # A corpus that annotates neither target is not a corpus this project can train on, and
    # the failure is worth catching at the loader rather than three steps later as an
    # all-NaN batch.
    if scheme == "none" and set(dataset["scale"]) == {"none"}:
        problems.append("a loader must annotate at least one target: scale and polarity_scheme are both 'none'")

    filled = sum(1 for value in dataset["polarity"] if not math.isnan(value))
    if filled:
        problems.append(
            f"polarity must be NaN in a loader ({filled} value(s) filled); harmonize.py owns that column"
        )


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
    _check_polarity(dataset, problems)

    if problems:
        raise ContractError("contract violated:\n  - " + "\n  - ".join(problems))
