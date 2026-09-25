"""Loader for CSMD v1, the reference corpus of the v2 merge.

CSMD is special among the v2 corpora: it is already on the ``da100`` scale that every
other corpus has to be mapped onto, so its ``label_raw`` passes through ``harmonize.py``
untouched. Every anchor-based mapping is fitted against this corpus.

The three published configs are merged and tagged through ``source``:

* ``meaning`` - 1355 pairs carrying a human meaning-preservation score;
* ``meaning_holdout_identical`` - 359 pairs of a sentence with itself, score 100;
* ``meaning_holdout_unrelated`` - 359 unrelated pairs, score 0.

The ``source`` tag is what lets the splitter keep the sanity-check pairs out of training,
which ``docs/H1-diagnostic-calibration.md`` shows v1 failed to do.
"""

from __future__ import annotations

from datasets import Dataset, concatenate_datasets, load_dataset

from data.schema import build

CORPUS: str = "csmd"
HF_REPO: str = "davebulaval/CSMD"
LICENSE: str = "MIT"

#: HF config name -> ``source`` tag of the contract.
CONFIGS: dict[str, str] = {
    "meaning": "original",
    "meaning_holdout_identical": "identical",
    "meaning_holdout_unrelated": "unrelated",
}


def _rows_from_config(config: str, source: str) -> list[dict]:
    """Load one CSMD config and turn it into contract rows.

    Args:
        config: HuggingFace config name.
        source: Contract ``source`` tag to attach to every row of this config.

    Returns:
        One dict per pair, ready for :func:`data.schema.build`.
    """
    dataset = load_dataset(HF_REPO, config)
    splits = sorted(dataset.keys())
    merged: Dataset = concatenate_datasets([dataset[split] for split in splits])
    split_of_row: list[str] = []
    for split in splits:
        # CSMD publishes 'train'/'dev'/'test'; the holdout configs only publish 'test'.
        hint = split if split in {"train", "dev", "test"} else ""
        split_of_row.extend([hint] * len(dataset[split]))

    return [
        {
            "item_id": f"{source}:{index}",
            "original": original,
            "simplification": simplification,
            "label_raw": float(label),
            "scale": "da100",
            # The published corpus reports averaged scores without per-annotator spread.
            "n_annotators": 0,
            "label_std": float("nan"),
            "source": source,
            "domain": "wiki",
            "system": "",
            "split_hint": hint,
            "license": LICENSE,
        }
        for index, (original, simplification, label, hint) in enumerate(
            zip(merged["original"], merged["simplification"], merged["label"], split_of_row)
        )
    ]


def load() -> Dataset:
    """Return CSMD v1 as a single contract-compliant dataset.

    Returns:
        The three configs merged, ``source``-tagged, on the ``da100`` scale with ``label``
        left as NaN like every other loader.
    """
    rows: list[dict] = []
    for config, source in CONFIGS.items():
        rows.extend(_rows_from_config(config, source))
    return build(rows, CORPUS)
