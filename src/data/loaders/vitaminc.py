"""VitaminC loader for CSMD v3: the bulk of the polarity signal.

488 904 claim-evidence pairs built from real Wikipedia revisions. What makes it the right
corpus here rather than merely the biggest: the pairs are **contrastive and minimal**. A
sentence was edited, and the claim that the previous revision supported is now refuted by
the next one. The two evidences differ by a date, a number, a name. That is exactly the
perturbation LexFlip showed current metrics cannot see, at a scale no hand-built probe
reaches.

It annotates polarity and **nothing else**: there is no human judgement of how much meaning
a pair shares, so ``scale`` is ``none`` and ``label_raw`` is NaN on every row. Declaring
the absence is the point. A default of ``0.0`` would read as "no meaning preserved", which
is the opposite of "not measured", and it would poison the magnitude head with half a
million confident zeroes.

Direction of the pair. The evidence is the premise and the claim is the hypothesis, so
``original`` is the evidence and ``simplification`` is the claim. Reversing them would ask
a different question: whether the evidence follows from the claim, which is not what the
annotators answered.

``revision_type`` distinguishes ``real`` pairs, taken from genuine Wikipedia edits, from
``synthetic`` ones written by annotators against an unchanged page. Both are kept and the
distinction is carried in ``system``, so a later experiment can test whether the synthetic
half behaves differently without reloading anything.
"""

from __future__ import annotations

from typing import Final

from datasets import Dataset

from data.schema import build

_DATASET: Final[str] = "tals/vitaminc"

#: The native label space, verified in the data on 2026-09-25. Strings, not integers, so
#: there is no encoding to get backwards here; the check is that nothing else appears.
_PERMITTED: Final[frozenset[str]] = frozenset({"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"})

_SPLIT_HINTS: Final[dict[str, str]] = {"train": "train", "validation": "dev", "test": "test"}

# Declared CC BY-SA 3.0, inherited from Wikipedia. Redistributable, but share-alike: any
# merged corpus built on it must carry a compatible licence. See docs/v3-echelle-signee.md
# for why the v3 merge goes out under CC BY-NC-SA 4.0.
_LICENSE: Final[str] = "CC-BY-SA-3.0"


class LabelSpaceError(RuntimeError):
    """Raised when the upstream label space stops matching what was verified."""


def _rows(records: list[dict], split: str) -> list[dict]:
    """Shape one split into contract rows.

    Args:
        records: Rows of one split of ``tals/vitaminc``.
        split: That split's HuggingFace name.

    Returns:
        Contract-shaped rows, one per record.

    Raises:
        LabelSpaceError: If a record carries a verdict outside the verified three. An
            unknown verdict is not a row to skip: it means the upstream label space moved,
            and every other row is then suspect too.
    """
    rows: list[dict] = []
    for record in records:
        verdict = str(record["label"]).strip()
        if verdict not in _PERMITTED:
            raise LabelSpaceError(
                f"{_DATASET}[{split}]: verdict {verdict!r} is outside the verified label space "
                f"{sorted(_PERMITTED)}; re-verify the mapping before loading anything"
            )

        evidence = str(record["evidence"]).strip()
        claim = str(record["claim"]).strip()
        if not evidence or not claim:
            # The contract forbids empty text on either side, and an empty evidence
            # carries no verdict worth learning from.
            continue

        rows.append(
            {
                "item_id": str(record["unique_id"]),
                "original": evidence,
                "simplification": claim,
                "label_raw": float("nan"),
                "scale": "none",
                "n_annotators": 1,
                "label_std": float("nan"),
                "source": "original",
                "domain": "wiki",
                # 'real' pairs come from genuine Wikipedia revisions, 'synthetic' ones were
                # written by annotators. Carried so the split can be studied later.
                "system": f"vitaminc-{record.get('revision_type', 'unknown')}",
                "split_hint": _SPLIT_HINTS[split],
                "license": _LICENSE,
                "polarity_raw": verdict,
                "polarity_scheme": "fact3",
            }
        )
    return rows


def load() -> Dataset:
    """Return VitaminC at the CSMD contract schema.

    Returns:
        A dataset with exactly the contract columns, ``scale`` fixed to ``none`` and
        ``polarity_scheme`` fixed to ``fact3``.

    Raises:
        LabelSpaceError: If an unknown verdict appears upstream.
    """
    from datasets import load_dataset

    data = load_dataset(_DATASET)
    rows: list[dict] = []
    for split in _SPLIT_HINTS:
        rows.extend(_rows(list(data[split]), split))
    return build(rows, corpus="vitaminc")
