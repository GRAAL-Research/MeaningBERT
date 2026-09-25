"""MoNLI loader for CSMD v3: downward monotonicity under negation.

1 202 minimal pairs that differ by one lexical substitution, inside or outside the scope of
a negation. "There is a man **not** wearing a hat" entails "There is a man not wearing a
sunhat", because negation flips the direction of entailment: under negation the more
specific term follows from the more general one, not the other way round. Outside a
negation the same substitution gives the opposite verdict.

This is the sharpest probe in the v3 set, and the cheapest. The substitution changes one
word; a metric driven by lexical overlap answers the same number on both halves by
construction, and there is no way for it to accidentally succeed.

**It carries no contradiction class.** Its two labels are ``entailment`` and ``neutral``,
so it opposes "follows from" to "does not follow from", which is a weaker and easier
question than "denies". That is a property of the corpus and not of the scheme, so it
declares ``nli3`` and emits only two of the three values. It is also why MoNLI is reported
separately in the dissociation diagnostic rather than averaged with SICK and NaN-NLI: a
meaning-preservation metric is *right* not to punish a neutral pair, which shares its
meaning, so a low amplitude here is not the failure a low amplitude on SICK is.
"""

from __future__ import annotations

from typing import Final

from datasets import Dataset

from data.schema import build

_DATASET: Final[str] = "tasksource/monli"

#: The native label space, verified in the data on 2026-09-25: strings, and only these two.
_PERMITTED: Final[frozenset[str]] = frozenset({"entailment", "neutral"})

_SPLIT_HINTS: Final[dict[str, str]] = {"train": "train", "test": "test"}

# No licence in the metadata. MoNLI is derived from SNLI, whose sentences come from Flickr
# captions; the derivation is published with the Geiger et al. paper. Unsettled here, which
# costs nothing: the v3 merge goes out under the strictest licence of its inputs.
_LICENSE: Final[str] = "unspecified (Geiger et al., MoNLI; derived from SNLI)"


class LabelSpaceError(RuntimeError):
    """Raised when the upstream label space stops matching what was verified."""


def _rows(records: list[dict], split: str) -> list[dict]:
    """Shape one split into contract rows.

    Args:
        records: Rows of one split of ``tasksource/monli``.
        split: That split's HuggingFace name.

    Returns:
        Contract-shaped rows, one per record.

    Raises:
        LabelSpaceError: If a record carries a label outside the verified two. A
            ``contradiction`` appearing upstream would not be a row to skip: it would mean
            the corpus gained a class, and the diagnostic's separate treatment of MoNLI
            would no longer be justified.
    """
    rows: list[dict] = []
    for index, record in enumerate(records):
        label = str(record["gold_label"]).strip()
        if label not in _PERMITTED:
            raise LabelSpaceError(
                f"{_DATASET}[{split}][{index}]: label {label!r} is outside the verified label space "
                f"{sorted(_PERMITTED)}; MoNLI is treated as carrying no contradiction class, and "
                "that assumption must be rechecked before loading anything"
            )

        rows.append(
            {
                "item_id": f"{split}-{index}",
                "original": record["sentence1"],
                "simplification": record["sentence2"],
                "label_raw": float("nan"),
                "scale": "none",
                "n_annotators": 1,
                "label_std": float("nan"),
                "source": "original",
                "domain": "mixed",
                "system": "human",
                "split_hint": _SPLIT_HINTS[split],
                "license": _LICENSE,
                "polarity_raw": label,
                "polarity_scheme": "nli3",
            }
        )
    return rows


def load() -> Dataset:
    """Return MoNLI at the CSMD contract schema.

    Returns:
        A dataset with exactly the contract columns, ``scale`` fixed to ``none`` and
        ``polarity_scheme`` fixed to ``nli3``, emitting only two of its three values.

    Raises:
        LabelSpaceError: If an unknown label appears upstream.
    """
    from datasets import load_dataset

    data = load_dataset(_DATASET)
    rows: list[dict] = []
    for split in _SPLIT_HINTS:
        rows.extend(_rows(list(data[split]), split))
    return build(rows, corpus="monli")
