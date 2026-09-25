"""PAWS loader for CSMD v3: the control that keeps the metric honest.

PAWS pairs have very high lexical overlap and are mostly **not** paraphrases, because they
were built by swapping words rather than by rewriting. "he secretly met the ambassador ...
asking for a passport to return to England through Scotland" against the same sentence
ending "to return to Scotland through England" shares nearly every token and says something
different.

That is the whole point. A metric that reports lexical overlap instead of meaning scores
these pairs high, and the two v2 sanity checks, identical at 100 and unrelated at 0, cannot
see the difference: both are satisfied by any monotone function of overlap, which is
precisely LexFlip's criticism. PAWS is where that failure becomes visible.

**Its negatives are not contradictions, and this loader refuses to call them that.** Two
sentences that are not paraphrases need not contradict each other: swapping two place names
produces a different claim, not a denial of the first. Mapping ``not_paraphrase`` onto
``contradiction`` at load time would destroy the control before it is ever used, and it
would teach the polarity head that "different" means "opposite". So the corpus keeps its
own two-value scheme, ``paraphrase2``, and ``harmonize.py`` decides what to do with it in
the open, where the decision can be argued with.

Label encoding, verified in the data on 2026-09-25 rather than read off the card, which
declares the classes as the uninformative names ``['0', '1']``: label 0 is the
England/Scotland swap above, so **0 is not a paraphrase and 1 is a paraphrase**.

Configuration ``labeled_final`` is the human-annotated Wikipedia portion, 65 401 pairs.
``labeled_swap`` and the unlabeled portions are machine-labelled and are deliberately left
out: a control built on noisy labels controls nothing.
"""

from __future__ import annotations

from typing import Final

from datasets import Dataset

from data.schema import build

_DATASET: Final[str] = "google-research-datasets/paws"
_CONFIG: Final[str] = "labeled_final"

#: Integer to named class. See the module docstring for the check behind it.
_POLARITY_BY_CODE: Final[dict[int, str]] = {0: "not_paraphrase", 1: "paraphrase"}

_SPLIT_HINTS: Final[dict[str, str]] = {"train": "train", "validation": "dev", "test": "test"}

# The PAWS repository publishes the data under the terms stated by Google Research; the
# HuggingFace metadata says only "other". The sentences come from Wikipedia. Treated as
# unsettled until read at the source, which costs nothing because the v3 merge goes out
# under the strictest licence of its inputs anyway.
_LICENSE: Final[str] = "other (google-research/paws; sentences from Wikipedia)"


class LabelSpaceError(RuntimeError):
    """Raised when the upstream label space stops matching what was verified."""


def _rows(records: list[dict], split: str) -> list[dict]:
    """Shape one split into contract rows.

    Args:
        records: Rows of one split of ``google-research-datasets/paws``.
        split: That split's HuggingFace name.

    Returns:
        Contract-shaped rows, one per record.

    Raises:
        LabelSpaceError: If a record carries a label outside ``{0, 1}``.
    """
    rows: list[dict] = []
    for record in records:
        code = int(record["label"])
        if code not in _POLARITY_BY_CODE:
            raise LabelSpaceError(
                f"{_DATASET}[{_CONFIG}][{split}]: label {code} is outside the verified encoding "
                f"{sorted(_POLARITY_BY_CODE)}; re-verify the mapping before loading anything"
            )

        rows.append(
            {
                "item_id": f"{split}-{record['id']}",
                "original": record["sentence1"],
                "simplification": record["sentence2"],
                "label_raw": float("nan"),
                "scale": "none",
                "n_annotators": 1,
                "label_std": float("nan"),
                "source": "original",
                "domain": "wiki",
                "system": "human",
                "split_hint": _SPLIT_HINTS[split],
                "license": _LICENSE,
                "polarity_raw": _POLARITY_BY_CODE[code],
                "polarity_scheme": "paraphrase2",
            }
        )
    return rows


def load() -> Dataset:
    """Return PAWS ``labeled_final`` at the CSMD contract schema.

    Returns:
        A dataset with exactly the contract columns, ``scale`` fixed to ``none`` and
        ``polarity_scheme`` fixed to ``paraphrase2``.

    Raises:
        LabelSpaceError: If an unknown label appears upstream.
    """
    from datasets import load_dataset

    data = load_dataset(_DATASET, _CONFIG)
    rows: list[dict] = []
    for split in _SPLIT_HINTS:
        rows.extend(_rows(list(data[split]), split))
    return build(rows, corpus="paws")
