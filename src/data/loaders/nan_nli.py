"""NaN-NLI loader for CSMD v3: sub-clausal negation, hand-built and tiny.

258 pairs, written by linguists to target negation that sits **below** the clause level:
"Not all people have had the opportunities you have had" against "Some people have not had
the opportunities you have had". The negation scopes over a quantifier rather than over the
whole proposition, which is where models that have merely learned "the word *not* means
contradiction" break.

It is too small to train on and is not meant to be trained on. It is a probe, and it is
already one of the three suites in ``src/diagnostics/dissociation.py``, where the published
v1 model scores an amplitude of -1.4 % and the v2 ``large`` model 18.1 %.

**Its single published split is named ``train``, and this loader does not propagate that
name.** The corpus imposes no split: it is one hand-built diagnostic set, and the ``train``
label is an artefact of how it was uploaded. Carrying it through would quietly let the
probe into the training data, and the probe would then measure nothing. The contract's
``split_hint`` is ``""`` for exactly this case, so that is what is emitted, and the decision
belongs to ``harmonize.py``.
"""

from __future__ import annotations

from typing import Final

from datasets import Dataset

from data.schema import build

_DATASET: Final[str] = "joey234/nan-nli"

#: The native label space, verified in the data on 2026-09-25. All three classes are
#: present: 117 contradiction, 97 entailment, 44 neutral.
_PERMITTED: Final[frozenset[str]] = frozenset({"entailment", "neutral", "contradiction"})

#: The one split the corpus publishes. Its name is an upload artefact, see the docstring.
_PUBLISHED_SPLIT: Final[str] = "train"

_LICENSE: Final[str] = "CC-BY-SA-4.0"


class LabelSpaceError(RuntimeError):
    """Raised when the upstream label space stops matching what was verified."""


def _rows(records: list[dict]) -> list[dict]:
    """Shape the corpus into contract rows.

    Args:
        records: Rows of ``joey234/nan-nli``.

    Returns:
        Contract-shaped rows, one per record, all with an empty ``split_hint``.

    Raises:
        LabelSpaceError: If a record carries a label outside the verified three.
    """
    rows: list[dict] = []
    for index, record in enumerate(records):
        label = str(record["label"]).strip()
        if label not in _PERMITTED:
            raise LabelSpaceError(
                f"{_DATASET}[{index}]: label {label!r} is outside the verified label space "
                f"{sorted(_PERMITTED)}; re-verify the mapping before loading anything"
            )

        rows.append(
            {
                "item_id": str(index),
                "original": record["premise"],
                "simplification": record["hypothesis"],
                "label_raw": float("nan"),
                "scale": "none",
                "n_annotators": 1,
                "label_std": float("nan"),
                "source": "original",
                "domain": "mixed",
                "system": "human",
                # Deliberately empty. See the module docstring: the published split name is
                # an upload artefact, and propagating it would let a probe into training.
                "split_hint": "",
                "license": _LICENSE,
                "polarity_raw": label,
                "polarity_scheme": "nli3",
            }
        )
    return rows


def load() -> Dataset:
    """Return NaN-NLI at the CSMD contract schema.

    Returns:
        A dataset with exactly the contract columns, ``scale`` fixed to ``none``,
        ``polarity_scheme`` fixed to ``nli3`` and ``split_hint`` empty throughout.

    Raises:
        LabelSpaceError: If an unknown label appears upstream.
    """
    from datasets import load_dataset

    return build(_rows(list(load_dataset(_DATASET)[_PUBLISHED_SPLIT])), corpus="nan_nli")
