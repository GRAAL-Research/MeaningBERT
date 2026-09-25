"""SimpleText CLEF 2025 error-taxonomy loader for CSMD v2.

Source: Vendeville, Ermakova & De Loor, "Resource for Error Analysis in Text
Simplification: New Taxonomy and Test Collection" (SIGIR 2025,
https://arxiv.org/abs/2505.16392). The taxonomy splits errors into four
categories: A. Fluency, B. Alignment, C. Information, D. Simplification.
Only C and D touch the *meaning* of the simplification; A and B are surface
form (grammar, prompt formatting) and are excluded here, per the mission
brief ("une erreur de lisibilite n'est pas une perte de sens").

Data provenance (see ``simpletext.report.json`` and ``RAPPORT.md`` for the
full account): the dataset's own repository,
https://github.com/bVendeville/Salted, briefly published two annotation CSVs
on 2025-02-18 and deleted them the same day, ahead of the "published freely
after the CLEF 2025 evaluation cycle" promise in the paper. That cycle has
since closed (CLEF 2025 Working Notes, September 2025) but the files were
never republished, in that repository or its ``simpletext-madics`` fork, as
of this loader's retrieval date. The two CSVs are still reachable as git
blobs at the pre-deletion commit, which this loader fetches by pinned commit
SHA for reproducibility. This is real human annotation data, not a
synthetic stand-in; the licensing status of the *data files themselves* is
unconfirmed (no LICENSE file in the repository), which is documented as an
open question for H4 rather than assumed.

The two files annotate the same underlying pool of (source sentence,
simplified sentence) pairs at different granularities:

- ``test_data.csv``: mostly one annotation per pair.
- ``inter_annotator_agreement.csv``: a subset annotated by up to five or six
  annotators, used in the source paper to measure agreement.

A handful of pairs were re-annotated by the *same* annotator (a
self-consistency check in the source study) and disagree with themselves.
These are collapsed by averaging before counting distinct annotators, so
``n_annotators`` reflects human annotators, not annotation events.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final
from urllib.request import urlopen

import pandas as pd
from datasets import Dataset

from data.schema import build

CORPUS: Final[str] = "simpletext"

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR: Final[Path] = _REPO_ROOT / "datastore" / "raw" / "simpletext"

# Commit predating the same-day deletion of the two CSVs from the Salted repository.
# Pinned so retrieval is reproducible even though the files no longer exist on `main`.
_PINNED_COMMIT: Final[str] = "242652d1a9e33f71e2eecd7b21ecf73c4f9eed81"
_RAW_FILES: Final[dict[str, str]] = {
    "test_data.csv": (f"https://raw.githubusercontent.com/bVendeville/Salted/{_PINNED_COMMIT}/test_data.csv"),
    "inter_annotator_agreement.csv": (
        f"https://raw.githubusercontent.com/bVendeville/Salted/{_PINNED_COMMIT}/inter_annotator_agreement.csv"
    ),
}

# C. Information + D. Simplification columns: the taxonomy branches that distort meaning.
# A. Fluency and B. Alignment (grammar, formatting) are deliberately excluded.
MEANING_ERROR_COLUMNS: Final[tuple[str, ...]] = (
    "C1. Factuality hallucination",
    "C2. Faithfulness hallucination",
    "C3. Topic shift",
    "D1.1. Overgeneralization",
    "D1.2 Overspecification of Concepts",
    "D2.1. Loss of Informative Content",
    "D2.2. Out-of-Scope Generation",
)


def _ensure_raw_files(raw_dir: Path) -> None:
    """Download the pinned CSVs into *raw_dir* if they are not already cached."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in _RAW_FILES.items():
        path = raw_dir / filename
        if path.exists():
            continue
        with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed, pinned https URL
            path.write_bytes(response.read())


def _read_raw(raw_dir: Path) -> pd.DataFrame:
    """Read and concatenate the two annotation CSVs from *raw_dir*.

    Args:
        raw_dir: Directory expected to hold both files listed in ``_RAW_FILES``.

    Returns:
        The row-level concatenation of both files (one row per annotation event).

    Raises:
        FileNotFoundError: If a required file is missing from *raw_dir*.
    """
    frames = []
    for filename in _RAW_FILES:
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"missing raw SimpleText file: {path}. Call load() with no arguments to "
                "auto-download the pinned CSVs, or populate the directory manually."
            )
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def _aggregate(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse annotation-event rows into one row per (run_id, snt_id) pair.

    Args:
        raw: Row-level annotations, one row per annotation event.

    Returns:
        A tuple of (aggregated per-pair frame, counts of rows dropped by reason).
    """
    missing_mask = raw["simplified sentence"].isna() | raw["source sentence"].isna()
    dropped = {"missing_text": int(missing_mask.sum())}
    frame = raw[~missing_mask].copy()

    frame["error_count"] = frame[list(MEANING_ERROR_COLUMNS)].sum(axis=1).astype(float)

    # Same annotator, same pair, disagreeing judgments (self-consistency reruns in the
    # source study): average them into one judgment before counting distinct annotators.
    per_annotator = frame.groupby(["run_id", "snt_id", "Annotator"], as_index=False).agg(
        original_text=("source sentence", "first"),
        simplified_text=("simplified sentence", "first"),
        error_count=("error_count", "mean"),
    )

    aggregated = per_annotator.groupby(["run_id", "snt_id"], as_index=False).agg(
        original_text=("original_text", "first"),
        simplified_text=("simplified_text", "first"),
        n_annotators=("Annotator", "nunique"),
        label_raw=("error_count", "mean"),
        label_std=("error_count", "std"),
    )
    aggregated.loc[aggregated["n_annotators"] < 2, "label_std"] = float("nan")
    return aggregated, dropped


def load(raw_dir: Path | None = None) -> Dataset:
    """Return the SimpleText error-taxonomy corpus at the CSMD v2 contract schema.

    Args:
        raw_dir: Directory holding the two source CSVs. Defaults to
            ``datastore/raw/simpletext/`` and triggers an automatic download of the
            pinned commit's files into that directory if they are not already present.
            Tests pass a fixtures directory here to avoid any network access.

    Returns:
        A dataset following ``src/data/schema.py``'s contract, with ``scale="error_count"``:
        the number of meaning-distorting errors (0 = fully preserved, higher = worse).
    """
    directory = raw_dir if raw_dir is not None else DEFAULT_RAW_DIR
    if raw_dir is None:
        _ensure_raw_files(directory)

    raw = _read_raw(directory)
    aggregated, _dropped = _aggregate(raw)

    rows = [
        {
            "item_id": f"{record['run_id']}//{record['snt_id']}",
            "original": record["original_text"],
            "simplification": record["simplified_text"],
            "label_raw": float(record["label_raw"]),
            "scale": "error_count",
            "n_annotators": int(record["n_annotators"]),
            "label_std": float(record["label_std"]) if math.isfinite(record["label_std"]) else float("nan"),
            "domain": "scientific",
            "source": "original",
            "system": str(record["run_id"]),
            "split_hint": "",
            "license": "unknown",
        }
        for record in aggregated.to_dict("records")
    ]
    return build(rows, corpus=CORPUS)
