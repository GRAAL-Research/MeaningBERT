"""SALSA loader for CSMD v2.

Status: **blocked before the contract boundary**. See ``RAPPORT.md`` at the repository
root, section "Question ouverte pour l'harmonisation", before touching :func:`load`.

Two independent problems, not one:

1. The full SALSA corpus (19K edit annotations on 840 simplifications, per the paper and
   ``README.md`` of https://github.com/davidheineman/salsa) was never published. The
   repository's own ``README.md`` and analysis notebooks reference a sibling ``data/``
   directory (``data/salsa_train.json``, ``data/salsa-non-adjudicated/...``) that does not
   exist anywhere in the repository, in any commit, on the one existing branch, in the one
   fork, or in any GitHub release. The only real SALSA annotations publicly reachable are
   the 50 sentence pairs bundled as the interactive-interface demo, served at
   ``https://thresh.tools/data/salsa.json`` (identical to ``interface/example_data.json``
   in the repository) and mirrored in the ``thresh`` tool's own ``public/data/salsa.json``.
   This loader downloads and works from that 50-pair demo subset. It is real, unmodified
   SALSA annotation data, not a fabrication, but it is a small fraction of the corpus the
   mission describes.
2. SALSA annotates **edits**, not sentence pairs. The CONTRACT wants one ``label_raw`` per
   sentence pair. The SALSA authors do have a sentence-level aggregation of their own
   (``calculate_sentence_score`` in ``analysis/utils/scoring.py``, the same target
   LENS-SALSA is trained to predict), but it (a) needs the raw per-batch annotation format
   that was never published either -- only the interface-demo format is available -- and
   (b) is an unbounded weighted sum with no fixed bounds, which does not match any ``scale``
   in ``CONTRACT.md``. Reusing it would mean silently inventing a bounded rescaling, which
   Rule 3 of ``BRIEF.md`` forbids. This loader therefore stops at the edit level, per the
   mission's own explicit fallback instruction, and :func:`load` raises
   :class:`SalsaAggregationPending` instead of guessing.

:func:`load_edit_level` is the actual deliverable: one row per (sentence pair, annotator,
edit), with the edit classified by SALSA's own content/syntax/lexical family logic (mirrored
here from ``analysis/utils/dataloader.py::process_annotation``, Apache-2.0). It is the
"fichier de travail" the mission asks for when no ready-made sentence score is usable.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import requests
from datasets import Dataset

logger = logging.getLogger(__name__)

CORPUS: Final[str] = "salsa"
RAW_DATA_URL: Final[str] = "https://thresh.tools/data/salsa.json"
DEFAULT_RAW_PATH: Final[Path] = Path("datastore/raw/salsa/salsa.json")
LICENSE: Final[str] = "Apache-2.0"
DOMAIN: Final[str] = "wiki"

LIKERT3: Final[dict[str, int]] = {"minor": 1, "somewhat": 2, "a lot": 3}
QUALITIES: Final[frozenset[str]] = frozenset({"good", "trivial", "bad"})
INFORMATION_IMPACTS: Final[frozenset[str]] = frozenset({"less", "more", "same", "different"})
FAMILIES: Final[frozenset[str]] = frozenset({"content", "syntax", "lexical"})
CATEGORIES: Final[frozenset[str]] = frozenset(
    {"deletion", "insertion", "substitution", "reorder", "split", "structure"}
)


class SalsaAggregationPending(RuntimeError):
    """Raised by :func:`load`: the edit-to-sentence aggregation is not this loader's call.

    See ``RAPPORT.md`` at the repository root before resolving this. Use
    :func:`load_edit_level` for the pre-aggregation working table in the meantime.
    """


class MalformedEdit(ValueError):
    """Raised internally when an edit dict does not match the documented SALSA schema."""


@dataclass(frozen=True)
class EditRecord:
    """One annotated edit, classified per SALSA's own edit-quality taxonomy.

    Attributes:
        pair_id: Stable id of the (original, simplification, system) triple, shared by
            every annotator who annotated that triple.
        item_id: ``pair_id`` plus the annotator, unique per annotation pass.
        original: Complex source sentence.
        simplification: Simplified candidate sentence.
        system: System that produced the simplification (``human`` or a model name).
        annotator: Anonymous annotator id from the source data.
        edit_index: Position of this edit within its annotation pass, for traceability.
        category: One of :data:`CATEGORIES`.
        subtype: Fine-grained edit type (e.g. ``good_deletion``, ``bad_deletion``).
        information_impact: One of :data:`INFORMATION_IMPACTS`.
        quality: One of :data:`QUALITIES`.
        severity: Likert-3 rating (1 minor, 2 moderate, 3 major) when applicable, else None.
        family: One of :data:`FAMILIES`, per SALSA's own content/syntax/lexical taxonomy.
        meaning_relevant: True if this edit type can affect meaning preservation, per the
            filter documented in ``RAPPORT.md``.
        grammar_error: Whether the edit was flagged as introducing a fluency error.
        coreference_error: Whether a deletion was flagged as breaking coreference.
        structure_type: Sub-type of a ``structure`` edit (voice, tense, ...), else None.
        reorder_level: ``word_level`` or ``component_level`` for a ``reorder`` edit.
    """

    pair_id: str
    item_id: str
    original: str
    simplification: str
    system: str
    annotator: str
    edit_index: int
    category: str
    subtype: str
    information_impact: str
    quality: str
    severity: int | None
    family: str
    meaning_relevant: bool
    grammar_error: bool
    coreference_error: bool
    structure_type: str | None
    reorder_level: str | None


# --- Raw data access ------------------------------------------------------------------


def _download_raw(destination: Path) -> None:
    """Download the SALSA interface-demo dataset to *destination*.

    Args:
        destination: File path to write the downloaded JSON to. Parent directories are
            created as needed.

    Raises:
        requests.RequestException: If the download fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading SALSA demo dataset from %s", RAW_DATA_URL)
    response = requests.get(RAW_DATA_URL, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)


def _load_raw(raw_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load the raw SALSA JSON, downloading it to the cache path if not already present.

    Args:
        raw_path: Path to a local SALSA JSON file. Defaults to :data:`DEFAULT_RAW_PATH`.
            Never fetched from the network when an explicit path is given, so tests can
            point this at a fixture without network access.

    Returns:
        The parsed list of annotation items, as served by the SALSA interface.
    """
    path = Path(raw_path) if raw_path is not None else DEFAULT_RAW_PATH
    if raw_path is None and not path.exists():
        _download_raw(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# --- Parsing helpers -------------------------------------------------------------------


def _unwrap(node: dict[str, Any]) -> tuple[str, Any]:
    """Unwrap one level of the SALSA interface's ``{val: name, name: rest}`` nesting.

    Args:
        node: A dict with a ``val`` key naming which sibling key holds the answer.

    Returns:
        The ``(name, rest)`` pair, where ``rest`` is ``None`` when the option has no
        follow-up question (e.g. a bare ``trivial`` leaf).

    Raises:
        MalformedEdit: If ``node`` has no ``val`` key.
    """
    try:
        name = node["val"]
    except (KeyError, TypeError) as error:
        raise MalformedEdit(f"expected a dict with a 'val' key, got {node!r}") from error
    return name, node.get(name)


def _severity_of(quality: str, rest: Any) -> int | None:
    """Resolve a Likert-3 severity from *rest*, or None for a quality with no rating."""
    if quality == "trivial":
        return None
    if not isinstance(rest, str) or rest not in LIKERT3:
        raise MalformedEdit(f"expected a likert-3 rating, got {rest!r}")
    return LIKERT3[rest]


def _parse_impact_leaf(node: dict[str, Any]) -> tuple[str, int | None]:
    """Parse a ``{val: bad|trivial|good, [bad|good]: likert}`` leaf shared by several edit types.

    Returns:
        The ``(quality, severity)`` pair.
    """
    quality, rest = _unwrap(node)
    if quality not in QUALITIES:
        raise MalformedEdit(f"unknown quality {quality!r}")
    return quality, _severity_of(quality, rest)


def _parse_deletion(annotation: dict[str, Any]) -> dict[str, Any]:
    subtype, rest = _unwrap(annotation["deletion_type"])
    quality = {"good_deletion": "good", "bad_deletion": "bad", "trivial_deletion": "trivial"}.get(subtype)
    if quality is None:
        raise MalformedEdit(f"unknown deletion subtype {subtype!r}")
    return {
        "subtype": subtype,
        "quality": quality,
        "severity": _severity_of(quality, rest),
        "information_impact": "less",
        "structure_type": None,
        "reorder_level": None,
        "coreference_error": annotation.get("coreference") == "yes",
    }


_INSERTION_ERROR_SUBTYPES: Final[frozenset[str]] = frozenset(
    {"repetition", "contradiction", "facutal_error", "irrelevant"}
)


def _parse_insertion_type(subtype: str, rest: Any) -> tuple[str, int | None]:
    """Shared by top-level ``insertion`` edits and the ``more`` branch of ``substitution``."""
    if subtype == "elaboration":
        return "good", _severity_of("good", rest)
    if subtype == "trivial_insertion":
        if isinstance(rest, dict):
            answer, sub_rest = _unwrap(rest)
        else:
            answer, sub_rest = rest, None
        if answer == "yes":
            return "good", _severity_of("good", sub_rest)
        if answer == "no":
            return "trivial", None
        raise MalformedEdit(f"unknown trivial_insertion answer {answer!r}")
    if subtype in _INSERTION_ERROR_SUBTYPES:
        return "bad", _severity_of("bad", rest)
    raise MalformedEdit(f"unknown insertion subtype {subtype!r}")


def _parse_insertion(annotation: dict[str, Any]) -> dict[str, Any]:
    subtype, rest = _unwrap(annotation["insertion_type"])
    quality, severity = _parse_insertion_type(subtype, rest)
    return {
        "subtype": subtype,
        "quality": quality,
        "severity": severity,
        "information_impact": "more",
        "structure_type": None,
        "reorder_level": None,
        "coreference_error": False,
    }


def _parse_substitution(annotation: dict[str, Any]) -> dict[str, Any]:
    impact, rest = _unwrap(annotation["substitution_info_change"])
    if impact not in INFORMATION_IMPACTS:
        raise MalformedEdit(f"unknown substitution information impact {impact!r}")

    if impact == "same":
        quality, severity = _parse_impact_leaf(rest)
        subtype = f"same:{quality}"
    elif impact == "less":
        deletion_subtype, deletion_rest = _unwrap(rest)
        quality = {"good_deletion": "good", "bad_deletion": "bad", "trivial_deletion": "trivial"}.get(deletion_subtype)
        if quality is None:
            raise MalformedEdit(f"unknown substitution/less subtype {deletion_subtype!r}")
        severity = _severity_of(quality, deletion_rest)
        subtype = deletion_subtype
    elif impact == "more":
        insertion_subtype, insertion_rest = _unwrap(rest)
        quality, severity = _parse_insertion_type(insertion_subtype, insertion_rest)
        subtype = insertion_subtype
    else:  # "different"
        # SALSA always treats a meaning-changing substitution as an error, regardless of
        # rating direction (mirrors process_diff_info in analysis/utils/dataloader.py,
        # which hard-codes Quality.ERROR for this branch).
        quality = "bad"
        severity = _severity_of("bad", rest)
        subtype = "different"

    return {
        "subtype": subtype,
        "quality": quality,
        "severity": severity,
        "information_impact": impact,
        "structure_type": None,
        "reorder_level": None,
        "coreference_error": False,
    }


def _parse_reorder(annotation: dict[str, Any]) -> dict[str, Any]:
    reorder_level, rest = _unwrap(annotation["reorder_level"])
    if reorder_level not in {"word_level", "component_level"}:
        raise MalformedEdit(f"unknown reorder level {reorder_level!r}")
    quality, severity = _parse_impact_leaf(rest)
    return {
        "subtype": f"{reorder_level}:{quality}",
        "quality": quality,
        "severity": severity,
        "information_impact": "same",
        "structure_type": None,
        "reorder_level": reorder_level,
        "coreference_error": False,
    }


def _parse_impact_only(annotation: dict[str, Any], category: str) -> dict[str, Any]:
    """Shared by ``split`` and the impact part of ``structure``."""
    quality, severity = _parse_impact_leaf(annotation["impact"])
    return {
        "subtype": f"{category}:{quality}",
        "quality": quality,
        "severity": severity,
        "information_impact": "same",
        "reorder_level": None,
        "coreference_error": False,
    }


def _parse_split(annotation: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse_impact_only(annotation, "split")
    parsed["structure_type"] = None
    return parsed


def _parse_structure(annotation: dict[str, Any]) -> dict[str, Any]:
    structure_type, _ = _unwrap(annotation["structure_type"])
    parsed = _parse_impact_only(annotation, "structure")
    parsed["subtype"] = structure_type
    parsed["structure_type"] = structure_type
    return parsed


_CATEGORY_PARSERS: Final[dict[str, Any]] = {
    "deletion": _parse_deletion,
    "insertion": _parse_insertion,
    "substitution": _parse_substitution,
    "reorder": _parse_reorder,
    "split": _parse_split,
    "structure": _parse_structure,
}


def classify_family(category: str, quality: str, information_impact: str) -> str:
    """Classify an edit into SALSA's own content/syntax/lexical family.

    Mirrors ``process_annotation`` in ``analysis/utils/dataloader.py`` (Apache-2.0,
    https://github.com/davidheineman/salsa): an edit that changes how much information is
    present is ``content``; a same-information substitution (a paraphrase) or any trivial
    edit is ``lexical``; anything else -- a non-trivial reorder, split or structure change
    that leaves the information amount alone -- is ``syntax``.

    Args:
        category: Edit category, one of :data:`CATEGORIES`.
        quality: One of :data:`QUALITIES`.
        information_impact: One of :data:`INFORMATION_IMPACTS`.

    Returns:
        One of :data:`FAMILIES`.
    """
    if information_impact != "same" and quality != "trivial":
        return "content"
    if category == "substitution" or quality == "trivial":
        return "lexical"
    return "syntax"


def _is_meaning_relevant(family: str, quality: str) -> bool:
    """Whether an edit can plausibly affect meaning preservation.

    ``content`` edits change the amount of information by definition: always relevant.
    ``lexical`` edits are relevant only when flagged ``bad`` (a paraphrase that keeps the
    same information amount but was judged harmful can still distort meaning; a trivial
    paraphrase or a good one cannot). ``syntax`` edits (reorder, split, structure) are
    fluency/simplicity concerns, not meaning-preservation ones, per the mission brief:
    "Une edition purement syntaxique qui preserve le sens ne doit pas penaliser le score de
    preservation" -- so they are never counted here, regardless of quality.
    """
    if family == "content":
        return True
    if family == "lexical":
        return quality == "bad"
    return False


def _pair_id(original: str, simplification: str, system: str) -> str:
    """Stable id for a (original, simplification, system) triple, independent of file order."""
    digest = hashlib.sha1(f"{original}\x1f{simplification}\x1f{system}".encode("utf-8")).hexdigest()
    return digest[:12]


def parse_item_edits(item: dict[str, Any], item_index: int) -> tuple[list[EditRecord], list[str]]:
    """Parse every edit of one annotated sentence pair.

    Args:
        item: One element of the raw SALSA JSON (``source``, ``target``, ``metadata``,
            ``edits``).
        item_index: Position of *item* in the raw file, used only in skip-reason messages.

    Returns:
        A ``(records, skip_reasons)`` pair. Malformed edits are skipped, not raised: one bad
        edit in a 19-edit sentence should not discard the other 18.
    """
    original = str(item.get("source", "")).strip()
    simplification = str(item.get("target", "")).strip()
    system = str(item.get("metadata", {}).get("system", ""))
    annotator = str(item.get("metadata", {}).get("annotator", ""))
    pair_id = _pair_id(original, simplification, system)
    item_id = f"{pair_id}:{annotator}"

    records: list[EditRecord] = []
    reasons: list[str] = []

    if not original or not simplification:
        reasons.append(f"item {item_index}: empty original or simplification, all edits skipped")
        return records, reasons

    for edit_index, edit in enumerate(item.get("edits", [])):
        category = edit.get("category")
        if category not in CATEGORIES:
            reasons.append(f"item {item_index} edit {edit_index}: unknown category {category!r}")
            continue
        try:
            parsed = _CATEGORY_PARSERS[category](edit.get("annotation", {}))
        except MalformedEdit as error:
            reasons.append(f"item {item_index} edit {edit_index} ({category}): {error}")
            continue

        family = classify_family(category, parsed["quality"], parsed["information_impact"])
        records.append(
            EditRecord(
                pair_id=pair_id,
                item_id=item_id,
                original=original,
                simplification=simplification,
                system=system,
                annotator=annotator,
                edit_index=edit_index,
                category=category,
                subtype=parsed["subtype"],
                information_impact=parsed["information_impact"],
                quality=parsed["quality"],
                severity=parsed["severity"],
                family=family,
                meaning_relevant=_is_meaning_relevant(family, parsed["quality"]),
                grammar_error=edit.get("annotation", {}).get("grammar_error") == "yes",
                coreference_error=parsed["coreference_error"],
                structure_type=parsed["structure_type"],
                reorder_level=parsed["reorder_level"],
            )
        )

    return records, reasons


# --- Public API --------------------------------------------------------------------------


def load_edit_level(raw_path: Path | str | None = None) -> Dataset:
    """Load SALSA at edit granularity: one row per (sentence pair, annotator, edit).

    This is the working table the mission's fallback branch asks for when no ready-made
    sentence-level target can be used. It is **not** CONTRACT-compliant: it has no
    ``label_raw``, no ``scale``, and several rows share the same sentence pair (one per
    annotator). See :func:`load` and the module docstring for why.

    Args:
        raw_path: Path to a local SALSA JSON file. Defaults to :data:`DEFAULT_RAW_PATH`,
            downloading it there if missing.

    Returns:
        A dataset with one row per edit and the fields of :class:`EditRecord`, plus a
        module-level count of skipped malformed edits recoverable via
        :func:`skip_reasons_for`.
    """
    raw_items = _load_raw(raw_path)
    rows: list[dict[str, Any]] = []
    for item_index, item in enumerate(raw_items):
        records, _ = parse_item_edits(item, item_index)
        rows.extend(asdict(record) for record in records)
    if not rows:
        return Dataset.from_dict({field: [] for field in EditRecord.__dataclass_fields__})
    columns: dict[str, list[Any]] = {field: [] for field in EditRecord.__dataclass_fields__}
    for row in rows:
        for field, value in row.items():
            columns[field].append(value)
    return Dataset.from_dict(columns)


def skip_reasons(raw_path: Path | str | None = None) -> list[str]:
    """Return every malformed-edit skip reason found while parsing the raw file.

    Args:
        raw_path: Same meaning as in :func:`load_edit_level`.

    Returns:
        One human-readable reason per skipped edit or empty sentence pair.
    """
    raw_items = _load_raw(raw_path)
    reasons: list[str] = []
    for item_index, item in enumerate(raw_items):
        _, item_reasons = parse_item_edits(item, item_index)
        reasons.extend(item_reasons)
    return reasons


def candidate_aggregations(records: list[EditRecord]) -> dict[str, float | None]:
    """Compute the mission's three candidate edit-to-sentence aggregations for one pair.

    These are illustrative only -- see the module docstring and ``RAPPORT.md``. None of
    them is applied by :func:`load`.

    Args:
        records: Edit records belonging to a single (sentence pair, annotator) instance.

    Returns:
        A dict with ``max_content_error_severity`` (candidate: worst single content error;
        None if there is none), ``content_error_count`` (candidate: how many content
        errors), and ``content_error_severity_sum`` (candidate: their severities added up).
    """
    content_errors = [r for r in records if r.family == "content" and r.quality == "bad"]
    severities = [r.severity for r in content_errors if r.severity is not None]
    return {
        "max_content_error_severity": max(severities) if severities else None,
        "content_error_count": float(len(content_errors)),
        "content_error_severity_sum": float(sum(severities)),
    }


def load() -> Dataset:
    """Not implemented: the edit-to-sentence aggregation is an open arbitration.

    Raises:
        SalsaAggregationPending: Always. Read ``RAPPORT.md`` at the repository root first.
    """
    raise SalsaAggregationPending(
        "SALSA has no usable ready-made sentence-level score (see module docstring) and "
        "this loader is not authorized to invent an edit-to-sentence aggregation on its "
        "own (BRIEF.md mission instructions). Use load_edit_level() for the pre-aggregation "
        "working table, and see RAPPORT.md > 'Question ouverte pour l'harmonisation' for "
        "the candidate aggregations awaiting a decision."
    )
