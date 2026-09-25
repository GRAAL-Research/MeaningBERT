"""Build the eight training corpora of the v2 experiment.

The experiment is a 4 x 2 factorial. Four corpus-and-split conditions form a ladder, each
step adding one correction, and each is trained with and without augmentation:

======  ==========================  ===================  ====================================
Cond.   Corpus                      Split                What the step isolates
======  ==========================  ===================  ====================================
``a``   CSMD v1                     row-level, as v1     reproduces the published number
``b``   CSMD v1                     grouped              the source-sentence leak (H5)
``c``   CSMD v1, H6 corrected       grouped              the permuted labels (H6)
``d``   v2, four corpora            grouped              what the new corpora actually add
======  ==========================  ===================  ====================================

Without ``a`` and ``b`` a gain measured at ``d`` cannot be attributed: it could be new
data, or the mere disappearance of a defect. That is correction L3 of
``docs/H5-fuite-par-phrase-source.md``.

Run on the training host, since back-translation needs torch::

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/data/build_corpus.py --output-dir data/v2
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Callable, Optional

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from data.augment import MODES, augment_splits
from data.corrections import apply_corrections
from data.harmonize import deduplicate, harmonize
from data.loaders import csmd, plaba, simpeval, simpletext
from data.schema import validate
from data.splits import assert_no_leakage, split_by_source_sentence

#: Corpora of the v2 condition. CSMD is the reference every scale is anchored against.
V2_LOADERS = {"csmd": csmd, "simpeval": simpeval, "simpletext": simpletext, "plaba": plaba}

#: Priority for cross-corpus deduplication. CSMD first because it is the reference and,
#: after the H6 correction, its labels for the shared pairs are the authoritative ones.
DEDUP_PRIORITY = ["csmd", "simpeval", "simpletext", "plaba"]

CONDITIONS = ("a", "b", "c", "d")


def _v1_row_split(dataset: Dataset, seed: int, dev_ratio: float = 0.1, test_ratio: float = 0.3) -> DatasetDict:
    """Reproduce v1's row-level stratified split, leak included.

    Kept verbatim so condition ``a`` measures what was actually published rather than a
    cleaned-up approximation of it.
    """
    indices = list(range(len(dataset)))
    strata = dataset["source"]
    train_dev, test = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=strata)
    train, dev = train_test_split(
        train_dev,
        test_size=dev_ratio / (1 - test_ratio),
        random_state=seed,
        stratify=[strata[i] for i in train_dev],
    )
    subset = dataset.select(sorted(test))
    return DatasetDict(
        {
            "train": dataset.select(sorted(train)),
            "dev": dataset.select(sorted(dev)),
            "test": subset,
            # v1 evaluated its sanity checks on rows drawn from the same pool.
            "sanity": subset.filter(lambda row: row["source"] in {"identical", "unrelated"}),
        }
    )


def build_condition(condition: str, seed: int) -> tuple[DatasetDict, dict]:
    """Build the splits for one condition of the ladder.

    Args:
        condition: One of ``a``, ``b``, ``c``, ``d``.
        seed: Split seed.

    Returns:
        The splits and a record of how they were built.

    Raises:
        ValueError: On an unknown condition.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition '{condition}'; expected one of {CONDITIONS}")

    corpora = {"csmd": csmd.load()} if condition in {"a", "b", "c"} else {n: m.load() for n, m in V2_LOADERS.items()}
    for dataset in corpora.values():
        validate(dataset)

    applied: dict[str, int] = {}
    if condition in {"c", "d"}:
        # The H6 correction needs SimpEval as the authoritative source for the shared pairs.
        with_authority = dict(corpora)
        if "simpeval" not in with_authority:
            with_authority["simpeval"] = simpeval.load()
        with_authority, applied = apply_corrections(with_authority)
        corpora = {name: with_authority[name] for name in corpora}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged, harmonisation = harmonize(corpora, reference="csmd")

    if condition == "a":
        # v1 deduplicated on the exact pair only, and split at the row level.
        splits = _v1_row_split(merged, seed=seed)
        dropped: dict[str, int] = {}
        split_record = {"kind": "v1 row-level, leak included"}
    else:
        merged, dropped = deduplicate(merged, priority=DEDUP_PRIORITY)
        splits, report = split_by_source_sentence(merged, seed=seed)
        assert_no_leakage(splits)
        split_record = {"kind": "grouped by source sentence", "report": report.summary()}

    record = {
        "condition": condition,
        "corpora": {name: len(dataset) for name, dataset in corpora.items()},
        "corrections": applied,
        "harmonisation": harmonisation.summary(),
        "cross_corpus_dropped": dropped,
        "split": split_record,
        "rows": {name: len(splits[name]) for name in splits},
    }
    return splits, record


def make_translator(batch_size: int = 128) -> Callable[[list[str]], list[str]]:
    """Build an English round-trip paraphraser through French.

    French is a pivot, not a target: nothing non-English enters the corpus. Loaded lazily
    so that every other part of this module runs without torch.

    Two details do most of the work on a Pascal card, where there is no bf16 and no flash
    attention to hide behind:

    * **length-sorted batching.** Batching in corpus order pads every sentence up to the
      longest in its batch, and these corpora mix 5-token and 80-token sentences. Sorting
      by length, translating, then restoring the original order cuts the padded compute
      several times over for an identical result.
    * **a generation budget tied to the input.** ``max_length=512`` makes the decoder
      willing to run far past anything a sentence-level corpus contains.
    """
    import torch  # pylint: disable=import-outside-toplevel
    from transformers import MarianMTModel, MarianTokenizer  # pylint: disable=import-outside-toplevel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = []
    for name in ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"):
        tokenizer = MarianTokenizer.from_pretrained(name)
        model = MarianMTModel.from_pretrained(name).to(device).eval()
        pairs.append((tokenizer, model))

    @torch.inference_mode()
    def hop(texts: list[str], tokenizer, model) -> list[str]:
        """Translate *texts* once, batching by length and restoring the input order."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list[Optional[str]] = [None] * len(texts)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            inputs = tokenizer(
                [texts[i] for i in chunk], return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)
            budget = int(inputs["input_ids"].shape[1] * 1.6) + 12
            decoded = tokenizer.batch_decode(
                model.generate(**inputs, max_length=budget, num_beams=1), skip_special_tokens=True
            )
            for index, text in zip(chunk, decoded):
                out[index] = text
        return [text or "" for text in out]

    def translate(texts: list[str]) -> list[str]:
        return hop(hop(texts, *pairs[0]), *pairs[1])

    return translate


def build_all(output_dir: str, seed: int, conditions: list[str], modes: list[str]) -> dict:
    """Build every requested corpus variant and write it to disk."""
    os.makedirs(output_dir, exist_ok=True)
    translator: Optional[Callable[[list[str]], list[str]]] = None
    manifest: dict[str, dict] = {}

    for condition in conditions:
        splits, record = build_condition(condition, seed=seed)
        for mode in modes:
            if mode == "full" and translator is None:
                print("Loading the back-translation models once...")
                translator = make_translator()
            augmented, counts = augment_splits(splits, mode, translate=translator, batch_size=4096, seed=seed)
            name = f"{condition}_{mode}"
            path = os.path.join(output_dir, name)
            augmented.save_to_disk(path)
            manifest[name] = {
                **record,
                "augmentation": {"mode": mode, "counts": counts},
                "final_rows": {split: len(augmented[split]) for split in augmented},
                "path": path,
            }
            print(f"  {name:10} train={len(augmented['train']):6} dev={len(augmented['dev']):5} "
                  f"test={len(augmented['test']):5} sanity={len(augmented['sanity']):5}")

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    """Build the eight corpora of the v2 experiment."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/v2", help="Where to write the variants.")
    parser.add_argument("--seed", type=int, default=42, help="Split and sampling seed.")
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES))
    args = parser.parse_args()

    manifest = build_all(args.output_dir, args.seed, args.conditions, args.modes)
    print(f"\n{len(manifest)} variant(s) written to {args.output_dir}")


if __name__ == "__main__":
    main()
