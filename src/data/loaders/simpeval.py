"""SimpEval loader for CSMD v2.

Two human-annotated subsets of the LENS repository (Yao-Dou/LENS, ACL 2023) are merged
into a single ``simpeval`` corpus:

- ``SimpEval_past`` (``simpeval_past.csv``): ~2 400 system simplifications of 100
  TurkCorpus/ASSET sentences, each rated 0-100 by 5 annotators on a single, undecomposed
  "overall quality" scale (adequacy, fluency and simplicity are not separated).
- ``SimpEval_2022`` (``simpDA_2022.csv``): 360 system simplifications of 60 fresh 2022
  Wikipedia sentences, each rated 0-100 by 3 annotators on three *separate* dimensions
  (adequacy, fluency, simplicity). Only ``Answer.adequacy`` is used here: it is the
  meaning-preservation axis the mission asks for. ``simpeval_2022.csv`` (the file used
  for the paper's headline metric-correlation table) reports a single undecomposed
  rating like ``simpeval_past`` and is deliberately not used, because ``simpDA_2022``
  gives us the adequacy axis directly on the same 360 pairs.

See ``RAPPORT.md`` at the repository root for the full reasoning, the CSMD-overlap
measurement and the open questions this raises for harmonization.
"""

from __future__ import annotations

import csv
import logging
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Final

from datasets import Dataset

from data.schema import build

LOGGER = logging.getLogger(__name__)

# Pinned to a commit, not a branch, so a re-run months from now sees the exact same bytes.
_LENS_COMMIT: Final[str] = "8fa5b149f099f0c75563ee8344a2411b0bffbc37"
_BASE_URL: Final[str] = f"https://raw.githubusercontent.com/Yao-Dou/LENS/{_LENS_COMMIT}/data"
_PAST_URL: Final[str] = f"{_BASE_URL}/simpeval_past.csv"
_DA2022_URL: Final[str] = f"{_BASE_URL}/simpDA_2022.csv"

_RAW_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "datastore" / "raw" / "simpeval"

# No LICENSE file covers Yao-Dou/LENS's ``data/`` folder (only ``lens/LICENSE`` for the
# metric checkpoint code, Apache-2.0). The underlying sentences come from ASSET /
# TurkCorpus (Wikipedia), whose own redistribution terms are not settled here either.
# Treated as non-redistributable until a licence-by-licence check (H4) says otherwise.
_LICENSE: Final[str] = "unspecified (no repo-level licence for LENS data/; upstream ASSET/TurkCorpus terms unchecked)"


def _clean(value: str | None) -> str:
    """Strip a CSV field, tolerating a short row where ``csv.DictReader`` yields ``None``."""
    return (value or "").strip()


def _download(url: str, destination: Path) -> Path:
    """Fetch *url* into *destination* unless it is already cached there.

    Args:
        url: Source URL, pinned to a commit.
        destination: Local cache path.

    Returns:
        *destination*, guaranteed to exist.
    """
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - pinned, https, read-only
        destination.write_bytes(response.read())
    return destination


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read *path* as a UTF-8 CSV with a header row."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _dedupe_within_corpus(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop rows whose ``(original, simplification, system)`` triple already appeared.

    Per ``CONTRACT.md``, a loader only dedupes inside its own corpus; cross-corpus
    duplicates (e.g. against CSMD itself) are ``harmonize.py``'s job, not this one's.

    Args:
        rows: Rows already shaped for :func:`data.schema.build`.

    Returns:
        The deduplicated rows, in original order, and how many were dropped.
    """
    seen: set[tuple[str, str, str]] = set()
    kept: list[dict] = []
    n_dropped = 0
    for row in rows:
        key = (row["original"], row["simplification"], row["system"])
        if key in seen:
            n_dropped += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, n_dropped


def _load_past_rows(path: Path) -> tuple[list[dict], dict[str, int]]:
    """Parse ``simpeval_past.csv`` into contract-shaped rows.

    Each source row already carries 5 annotator ratings (``rating_1``..``rating_5``) on
    a single 0-100 "overall quality" scale for one (original, system) pair, so no
    cross-row aggregation is needed here (unlike the 2022 subset).

    Args:
        path: Local path to ``simpeval_past.csv``.

    Returns:
        The parsed rows, and a small counter dict of rows dropped and why.
    """
    rows: list[dict] = []
    counters = {"empty_simplification": 0, "malformed_ratings": 0}

    for record in _read_csv(path):
        original = _clean(record.get("original"))
        # ``processed_generation`` is the true-cased, cleaned-up form of the raw model
        # output (``generation``); it is what a human would actually read, so it is what
        # becomes ``simplification`` here.
        simplification = _clean(record.get("processed_generation"))
        if not simplification:
            counters["empty_simplification"] += 1
            continue

        try:
            ratings = [float(record[f"rating_{i}"]) for i in range(1, 6)]
        except (KeyError, TypeError, ValueError):
            counters["malformed_ratings"] += 1
            continue

        rows.append(
            {
                "item_id": f"past-{record['id']}",
                "original": original,
                "simplification": simplification,
                "label_raw": statistics.mean(ratings),
                "scale": "da100",
                "n_annotators": len(ratings),
                "label_std": statistics.stdev(ratings),
                "source": "identical" if original == simplification else "original",
                "domain": "wiki",
                "system": record.get("system", ""),
                "split_hint": "",
                "license": _LICENSE,
            }
        )
    return rows, counters


def _load_2022_rows(path: Path) -> tuple[list[dict], dict[str, int]]:
    """Parse ``simpDA_2022.csv`` into contract-shaped rows, aggregating over annotators.

    The source file has one row per (pair, worker): 3 workers rated each of the 360
    (original, system) pairs on adequacy, fluency and simplicity separately. This
    aggregates ``Answer.adequacy`` across the workers of each pair.

    Args:
        path: Local path to ``simpDA_2022.csv``.

    Returns:
        The parsed rows, and a small counter dict of rows dropped and why.
    """
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    counters = {"empty_text": 0, "malformed_adequacy": 0}

    for record in _read_csv(path):
        key = (
            record.get("Input.id", ""),
            record.get("Input.system", ""),
            _clean(record.get("Input.original")),
            _clean(record.get("Input.simplified")),
        )
        groups[key].append(record)

    rows: list[dict] = []
    for (item_id, system, original, simplification), members in groups.items():
        if not original or not simplification:
            counters["empty_text"] += len(members)
            continue

        try:
            adequacy = [float(member["Answer.adequacy"]) for member in members]
        except (KeyError, TypeError, ValueError):
            counters["malformed_adequacy"] += len(members)
            continue

        system_slug = system.replace(" ", "_") if system else "unknown"
        rows.append(
            {
                "item_id": f"2022-{item_id}-{system_slug}",
                "original": original,
                "simplification": simplification,
                "label_raw": statistics.mean(adequacy),
                "scale": "da100",
                "n_annotators": len(adequacy),
                "label_std": statistics.stdev(adequacy) if len(adequacy) > 1 else float("nan"),
                "source": "identical" if original == simplification else "original",
                "domain": "wiki",
                "system": system,
                "split_hint": "",
                "license": _LICENSE,
            }
        )
    return rows, counters


def load() -> Dataset:
    """Return SimpEval (past + 2022 adequacy) at the CSMD v2 contract schema.

    Returns:
        A dataset with exactly the contract columns, ``label`` all NaN, ``scale`` fixed
        to ``da100``, ``corpus`` fixed to ``simpeval``.
    """
    past_path = _download(_PAST_URL, _RAW_DIR / "simpeval_past.csv")
    da2022_path = _download(_DA2022_URL, _RAW_DIR / "simpDA_2022.csv")

    past_rows, _ = _load_past_rows(past_path)
    da2022_rows, _ = _load_2022_rows(da2022_path)

    rows, _ = _dedupe_within_corpus(past_rows + da2022_rows)
    return build(rows, corpus="simpeval")
