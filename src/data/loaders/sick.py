"""SICK loader for CSMD v3: the bridge between the two heads.

SICK is the only corpus in the v3 campaign that annotates **both** targets on **the same
pairs**: a continuous relatedness score and a three-way inference label. Every other
corpus answers one question or the other. That is why the composition
``signe = magnitude x (1 - 2 p_contradiction)`` is calibrated here and nowhere else: only
here can we ask what relatedness a human gave to a pair they also called a contradiction.

**No single English copy of SICK opens any more.** The canonical dataset still ships a
loading script, which HuggingFace dropped. The two halves survive as separate datasets
over the same pairs, so this loader joins them back together on the sentence pair itself:

- ``yangwang825/sick`` carries the inference label, with the original train / validation /
  test split (4439 / 495 / 4906);
- ``mteb/sickr-sts`` carries the relatedness score, published as one undivided split of
  9927 rows.

The join is exact: measured on 2026-09-25, **100.0 % of the rows of all three splits find
their relatedness score**, so nothing is dropped and no split is diluted. A future upstream
edit could break that, so :func:`_join` counts the misses and :func:`load` refuses to
return a corpus that lost more than a hundredth of itself rather than silently shrinking.

Label encoding, verified in the data and not read off a card. ``yangwang825/sick`` encodes
its classes as the integers 0, 1 and 2, and no dataset card says which is which. Two
independent checks agree:

1. the first pair of the test split, "A group of kids is playing in a yard ..." against
   "A group of boys in a yard is playing ...", is labelled 1, and it is plainly neutral;
2. the class priors match published SICK: label 1 is 50 % of the train split, label 0 is
   27 % and label 2 is 14 %, against the documented majority-neutral, then entailment,
   then contradiction ordering.

Getting this backwards is silent. It already happened once on this project, in
``src/diagnostics/dissociation.py``, where the swap surfaced only as an AUC of 0.021.
"""

from __future__ import annotations

import logging
from typing import Final

from datasets import Dataset

from data.schema import build

LOGGER = logging.getLogger(__name__)

_ENTAILMENT_DATASET: Final[str] = "yangwang825/sick"
_RELATEDNESS_DATASET: Final[str] = "mteb/sickr-sts"

#: Integer to named class. See the module docstring for the two checks behind it.
_POLARITY_BY_CODE: Final[dict[int, str]] = {0: "entailment", 1: "neutral", 2: "contradiction"}

#: HuggingFace split name to the contract's ``split_hint`` vocabulary.
_SPLIT_HINTS: Final[dict[str, str]] = {"train": "train", "validation": "dev", "test": "test"}

# Neither half declares a licence in its metadata. SICK itself is distributed by the
# SemEval-2014 Task 1 organisers, and the relatedness copy that this loader joins against
# is published as CC BY-NC-SA 3.0. The strictest of the two governs, and the corpus is
# treated as non-redistributable until checked pair by pair.
_LICENSE: Final[str] = "unspecified (SemEval-2014 Task 1; mteb/sickr-sts published as CC BY-NC-SA 3.0)"

#: Refuse to return the corpus if the join loses more than this share of the rows. The
#: measured loss is 0.0; anything above a hundredth means an upstream edit moved the text,
#: and a quietly smaller bridge is worse than a loud failure.
_MAX_UNJOINED_SHARE: Final[float] = 0.01


class JoinError(RuntimeError):
    """Raised when the two halves of SICK stop lining up."""


def _key(left: str, right: str) -> tuple[str, str]:
    """Join key for one pair: both sentences, stripped, in order.

    Order matters and is deliberately kept: SICK publishes A/B and B/A as distinct rows
    with distinct labels, so normalising the order would merge rows that disagree.
    """
    return (left.strip(), right.strip())


def _relatedness_index(records: list[dict]) -> dict[tuple[str, str], float]:
    """Index the relatedness half by sentence pair.

    Args:
        records: Rows of ``mteb/sickr-sts``, each with ``sentence1``, ``sentence2``,
            ``score``.

    Returns:
        Pair to relatedness score, on the native 1-5 scale.
    """
    return {_key(record["sentence1"], record["sentence2"]): float(record["score"]) for record in records}


def _join(
    entailment_records: list[dict],
    split: str,
    relatedness: dict[tuple[str, str], float],
) -> tuple[list[dict], int]:
    """Shape one split of the entailment half into contract rows, attaching relatedness.

    Args:
        entailment_records: Rows of one split of ``yangwang825/sick``.
        split: That split's HuggingFace name.
        relatedness: Output of :func:`_relatedness_index`.

    Returns:
        The contract-shaped rows, and how many records found no relatedness score.

    Raises:
        JoinError: If a record carries a label outside ``{0, 1, 2}``. An unknown code is
            not a row to skip: it means the upstream encoding changed, and every other
            row's polarity is then suspect too.
    """
    rows: list[dict] = []
    unjoined = 0

    for index, record in enumerate(entailment_records):
        code = int(record["label"])
        if code not in _POLARITY_BY_CODE:
            raise JoinError(
                f"{_ENTAILMENT_DATASET}[{split}][{index}]: label {code} is outside the verified "
                f"encoding {sorted(_POLARITY_BY_CODE)}; the upstream label space changed and the "
                "mapping must be re-verified before anything is loaded"
            )

        key = _key(record["text1"], record["text2"])
        score = relatedness.get(key)
        if score is None:
            unjoined += 1
            continue

        rows.append(
            {
                "item_id": f"{split}-{index}",
                "original": record["text1"],
                "simplification": record["text2"],
                "label_raw": score,
                "scale": "likert5",
                # SICK relatedness is published as a mean over 10 crowd annotators, but
                # the per-annotator ratings are not in either copy, so the dispersion is
                # unrecoverable here and must not be invented.
                "n_annotators": 10,
                "label_std": float("nan"),
                "source": "original",
                # Flickr8k captions plus SemEval video descriptions: neither wiki nor news.
                "domain": "mixed",
                "system": "human",
                "split_hint": _SPLIT_HINTS[split],
                "license": _LICENSE,
                "polarity_raw": _POLARITY_BY_CODE[code],
                "polarity_scheme": "nli3",
            }
        )

    return rows, unjoined


def check_join_loss(unjoined: int, total: int) -> None:
    """Refuse a bridge that quietly lost rows.

    Args:
        unjoined: Pairs that found no relatedness score.
        total: Pairs seen across all three splits.

    Raises:
        JoinError: If the loss exceeds :data:`_MAX_UNJOINED_SHARE`. A smaller SICK is not
            a smaller inconvenience: it is the only corpus carrying both targets, so a
            silent shrink biases the calibration of the whole composition.
    """
    if unjoined > _MAX_UNJOINED_SHARE * total:
        raise JoinError(
            f"{unjoined}/{total} SICK pairs found no relatedness score in {_RELATEDNESS_DATASET}; "
            "the two halves no longer line up and the bridge cannot be trusted"
        )
    if unjoined:
        LOGGER.warning("SICK: %d/%d pairs joined no relatedness score and were dropped", unjoined, total)


def load() -> Dataset:
    """Return SICK, both halves joined, at the CSMD contract schema.

    Returns:
        A dataset with exactly the contract columns, ``label`` and ``polarity`` all NaN,
        ``scale`` fixed to ``likert5``, ``polarity_scheme`` fixed to ``nli3``.

    Raises:
        JoinError: If more than :data:`_MAX_UNJOINED_SHARE` of the rows lose their
            relatedness score, or if an unknown inference label appears.
    """
    from datasets import load_dataset

    entailment = load_dataset(_ENTAILMENT_DATASET)
    relatedness = _relatedness_index(list(load_dataset(_RELATEDNESS_DATASET)["test"]))

    rows: list[dict] = []
    unjoined = 0
    total = 0
    for split in _SPLIT_HINTS:
        records = list(entailment[split])
        total += len(records)
        split_rows, split_unjoined = _join(records, split, relatedness)
        rows.extend(split_rows)
        unjoined += split_unjoined

    check_join_loss(unjoined, total)
    return build(rows, corpus="sick")
