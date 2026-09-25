"""Known defects in the source corpora, and the evidence for each.

A correction here is never a preference. Each one states what is wrong, how it was
established, and what the fix does. If the evidence is not reproducible, the correction
does not belong in this module.

See ``docs/H6-etiquettes-permutees-csmd.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from datasets import Dataset

from data.harmonize import pair_key

#: Corpus whose labels are authoritative for the pairs CSMD imported from SimpDA_2022.
SIMPDA_AUTHORITY: str = "simpeval"


@dataclass
class Correction:
    """One documented corpus defect and its fix."""

    name: str
    corpus: str
    evidence: str
    apply: Callable[[dict[str, Dataset]], tuple[dict[str, Dataset], int]]


def _csmd_simpda_relabel(corpora: dict[str, Dataset]) -> tuple[dict[str, Dataset], int]:
    """Replace CSMD's SimpDA_2022 labels with the ones from the authoritative source.

    CSMD's 360 SimpDA_2022 rows hold exactly the right multiset of adequacy scores, but
    attached to the wrong sentence pairs. The source CSV carries the pair and its score on
    the same line, so it cannot be misaligned; CSMD's copy can, and is.

    Only pairs that both corpora contain are touched, and only their ``label_raw``. Nothing
    is dropped, no text is altered, and CSMD keeps its own labels everywhere else.
    """
    if SIMPDA_AUTHORITY not in corpora or "csmd" not in corpora:
        return corpora, 0

    authority = corpora[SIMPDA_AUTHORITY]
    truth: dict[str, list[float]] = {}
    for original, simplification, item_id, value in zip(
        authority["original"], authority["simplification"], authority["item_id"], authority["label_raw"]
    ):
        # Only the 2022 subset is implicated; the `past` subset agrees with CSMD at r=0.68.
        if ":2022-" in item_id:
            truth.setdefault(pair_key(original, simplification), []).append(float(value))

    csmd = corpora["csmd"]
    replaced = 0
    labels: list[float] = []
    for original, simplification, value in zip(csmd["original"], csmd["simplification"], csmd["label_raw"]):
        key = pair_key(original, simplification)
        if key in truth:
            labels.append(sum(truth[key]) / len(truth[key]))
            replaced += 1
        else:
            labels.append(float(value))

    fixed = dict(corpora)
    fixed["csmd"] = csmd.remove_columns(["label_raw"]).add_column("label_raw", labels)
    return fixed, replaced


CORRECTIONS: list[Correction] = [
    Correction(
        name="csmd-simpda2022-relabel",
        corpus="csmd",
        evidence=(
            "The 360 pairs CSMD shares with SimpDA_2022 hold an identical multiset of "
            "adequacy scores (360/360 match, max difference 0.000000) yet correlate at "
            "r=0.009. Identical marginals with no correlation is a permutation, not a "
            "disagreement between annotation protocols. The permutation is scrambled both "
            "within and across source sentences, so it is not a sort-order artefact. "
            "CSMD's other sources are unaffected: its 127 pairs shared with simpeval_past "
            "correlate at r=0.683. See docs/H6-etiquettes-permutees-csmd.md."
        ),
        apply=_csmd_simpda_relabel,
    )
]


def apply_corrections(corpora: dict[str, Dataset]) -> tuple[dict[str, Dataset], dict[str, int]]:
    """Apply every known correction, returning the corrected corpora and what each changed.

    Args:
        corpora: Loader outputs keyed by corpus name.

    Returns:
        The corrected corpora, and how many rows each correction touched.
    """
    applied: dict[str, int] = {}
    for correction in CORRECTIONS:
        corpora, count = correction.apply(corpora)
        applied[correction.name] = count
    return corpora, applied
