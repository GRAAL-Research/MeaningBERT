"""PLABA loader for CSMD v2.

Source and scope
-----------------
PLABA (Attal et al. 2023, Nature Scientific Data, CC-BY-4.0) is a parallel corpus of
750 PubMed abstracts adapted phrase by phrase into plain language by biomedical experts.
That parallel corpus alone carries **no human preservation-of-meaning score**: it is a set
of adaptations, not a set of judgments. Per ``BRIEF.md``, a parallel corpus without a human
score does not enter CSMD v2, so this loader does not touch it.

The score we need instead comes from the TREC PLABA track (2023/2024), where biomedical
experts manually rated system outputs on four axes: Simplicity, Accuracy, Completeness,
Brevity (Ondov et al., "Lessons from the TREC Plain Language Adaptation of Biomedical
Abstracts (PLABA) track", 2025, https://pubmed.ncbi.nlm.nih.gov/40969486/). The official
per-sentence judgment files live behind TREC's participant-only results area
(``https://trec.nist.gov/results/...`` returns HTTP 401 without conference credentials),
so they are not retrievable here.

A partial mirror of that same manual evaluation exists in a co-author's own repository,
``https://github.com/ondovb/plaba-ft`` (commit history current as of 2026-09-19), under
``eval/manual/*.acc.csv``. Each file holds one evaluated system's sentence-level Accuracy
judgments on the TREC 2023 round-1 protocol: one row per (abstract, sentence), with the
source sentence, the system output, and two scores, ``Acc. comp.`` (Completeness -- how
much of the source information survives) and ``Acc. faith.`` (Faithfulness -- do the
output's points match the source's). This loader reads those files (see ``RAPPORT.md`` for
how to obtain them; they are not committed, per the environment rules).

Scale
-----
Both scores are recorded on a symmetric 3-point Likert scale, ``{-1, 0, 1}`` (higher is
better), confirmed against the source paper's own description of the annotation protocol.
This originally had no match in ``src/data/CONTRACT.md``'s ``SCALES`` table (``severity3``
shares the cardinality but not the bounds or the orientation, and reusing it would have
silently inverted the signal). The contract owner has since added ``likert3_signed``
(``low=-1.0``, ``high=1.0``, ``higher_is_better=True``) to that table for exactly this case;
see ``CONTRACT.md`` and ``RAPPORT.md`` for the decision trail. ``load()`` now emits
``scale="likert3_signed"``.

Axis choice: this loader targets **Faithfulness** ("Do points made in the output match
those of the source?"), which is the closer analogue to CSMD's "preservation du sens" than
Completeness ("how much of the source survives"), since a simplification can legitimately
omit detail while still being faithful to what it does say. Completeness is captured during
parsing (see ``open_question_completeness_raw`` in :func:`extract_rows`) but not emitted;
it is left as an open question in ``RAPPORT.md``.

Vigilance point (H2): the emitted ``label_raw`` is heavily concentrated near the upper
bound (mean 0.9417 on 806 rows -- see ``plaba.report.json`` and ``RAPPORT.md``), because the
evaluated systems are strong modern baselines and the human rows are expert references.
Whoever tests H2 should not read a high correlation on this corpus alone as a strong
signal: with almost all labels near +1, there is little variance for a metric to explain.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from datasets import Dataset

from data.schema import build

RAW_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "datastore" / "raw" / "plaba" / "eval" / "manual"

CORPUS: Final[str] = "plaba"
DOMAIN: Final[str] = "biomedical"
SCALE: Final[str] = "likert3_signed"
LABEL_COLUMN: Final[str] = "Acc. faith."
OPEN_QUESTION_COLUMN: Final[str] = "Acc. comp."
LICENSE: Final[str] = "unspecified (source repository ships no LICENSE file)"

# Filename stem (without ".acc.csv") -> (system value, is_human_reference).
_SYSTEM_BY_STEM: Final[dict[str, tuple[str, bool]]] = {
    "GPT-3.5-zero_shot": ("gpt-3.5-zero-shot", False),
    "Llama-2-7B-chat": ("llama-2-7b-chat", False),
    "Llama-2-7B-chat-SCER-0.5": ("llama-2-7b-chat-scer-0.5", False),
    "Manual-1": ("human", True),
    "Manual-2": ("human", True),
    "Manual-3": ("human", True),
    "Manual-4": ("human", True),
}


class PlabaSourceFormatError(RuntimeError):
    """Raised when a raw PLABA file does not match the expected structure.

    This means the upstream source changed shape underneath us; guessing a new mapping
    would be fabricating structure, not just data, so this loader refuses instead.
    """


@dataclass
class ExtractionResult:
    """Intermediate, pre-contract rows plus bookkeeping for the delivery report."""

    rows: list[dict] = field(default_factory=list)
    n_rows_read: int = 0
    n_rows_dropped_empty_text: int = 0
    n_rows_dropped_non_finite_label: int = 0
    n_duplicates_dropped: int = 0
    systems_seen: set[str] = field(default_factory=set)


def _parse_score(raw: str) -> float:
    """Parse a PLABA manual-judgment cell, tolerating the ``-1``/``-1.0`` mix seen in the wild.

    Args:
        raw: Raw CSV cell value.

    Returns:
        The score as a float, or ``nan`` if the cell is blank or not a number.
    """
    raw = raw.strip()
    if not raw:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _iter_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a PLABA ``*.acc.csv`` file into a list of raw string dicts.

    Args:
        path: Path to one system's accuracy CSV, formatted
            ``Abst,Sent,Source,Output,Acc. comp.,Acc. faith.``.

    Returns:
        One dict per data row, keyed by header name.

    Raises:
        PlabaSourceFormatError: If the file is missing the expected header columns.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"Abst", "Sent", "Source", "Output", "Acc. comp.", "Acc. faith."}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise PlabaSourceFormatError(
                f"{path}: expected columns {sorted(expected)}, found {reader.fieldnames}. "
                "Refusing to guess a mapping for a changed source format."
            )
        return list(reader)


def extract_rows(raw_dir: Path = RAW_DIR) -> ExtractionResult:
    """Parse every ``*.acc.csv`` file under ``raw_dir`` into contract-shaped, pre-build rows.

    A row is dropped when the source or output text is empty after stripping (6 real rows in
    the 2026-09-19 mirror: sentences the annotators judged as effectively omitted, scored -1
    on both axes with an empty ``Output`` cell) or when the faithfulness score itself is
    missing or non-numeric. Rows are deduplicated within the corpus on
    ``(original, simplification, system)``, per CONTRACT.md's deduplication rule.

    Args:
        raw_dir: Directory holding the per-system ``*.acc.csv`` files. Defaults to
            ``datastore/raw/plaba/eval/manual``.

    Returns:
        The parsed rows (native -1..1 scale, not yet contract-built) plus counters
        describing what was read, kept and dropped.

    Raises:
        FileNotFoundError: If ``raw_dir`` does not exist, i.e. the raw data was never
            fetched (see RAPPORT.md for the retrieval steps).
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"{raw_dir} does not exist. Fetch the manual-evaluation CSVs first; see RAPPORT.md.")

    result = ExtractionResult()
    seen_keys: set[tuple[str, str, str]] = set()

    for path in sorted(raw_dir.glob("*.acc.csv")):
        stem = path.name.removesuffix(".acc.csv")
        if stem not in _SYSTEM_BY_STEM:
            raise PlabaSourceFormatError(f"{path}: unrecognised system file stem {stem!r}, refusing to guess.")
        system, _is_human = _SYSTEM_BY_STEM[stem]
        slug = stem.lower()

        for csv_row in _iter_csv_rows(path):
            result.n_rows_read += 1
            original = csv_row["Source"].strip()
            simplification = csv_row["Output"].strip()
            if not original or not simplification:
                result.n_rows_dropped_empty_text += 1
                continue

            label_raw = _parse_score(csv_row[LABEL_COLUMN])
            if not math.isfinite(label_raw):
                result.n_rows_dropped_non_finite_label += 1
                continue

            dedup_key = (original, simplification, system)
            if dedup_key in seen_keys:
                result.n_duplicates_dropped += 1
                continue
            seen_keys.add(dedup_key)

            result.systems_seen.add(system)
            result.rows.append(
                {
                    "item_id": f"{slug}:{csv_row['Abst']}:{csv_row['Sent']}",
                    "original": original,
                    "simplification": simplification,
                    "label_raw": label_raw,
                    "open_question_completeness_raw": _parse_score(csv_row[OPEN_QUESTION_COLUMN]),
                    "n_annotators": 1,
                    "label_std": float("nan"),
                    "domain": DOMAIN,
                    "system": system,
                    "split_hint": "",
                }
            )

    return result


def load() -> Dataset:
    """Return the CONTRACT-compliant PLABA dataset (Faithfulness axis, ``likert3_signed``).

    Returns:
        One row per (source sentence, system output) pair evaluated for Faithfulness by
        TREC PLABA expert annotators, on the ``likert3_signed`` scale.

    Raises:
        FileNotFoundError: If the raw CSVs have not been fetched into
            ``datastore/raw/plaba/eval/manual``.
        PlabaSourceFormatError: If a raw file does not match the expected structure.
    """
    result = extract_rows(RAW_DIR)
    rows = [
        {
            "item_id": row["item_id"],
            "original": row["original"],
            "simplification": row["simplification"],
            "label_raw": row["label_raw"],
            "scale": SCALE,
            "n_annotators": row["n_annotators"],
            "label_std": row["label_std"],
            "domain": row["domain"],
            "system": row["system"],
            "split_hint": row["split_hint"],
            "license": LICENSE,
        }
        for row in result.rows
    ]
    return build(rows, CORPUS)
