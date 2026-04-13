"""Pre-generate all dataset variants to disk for fast training.

Run once before sweeps::

    python prepare_datasets.py --output_dir ./data

This creates::

    data/
        folds/
            fold_0/                             - seed 42, stratified split
                meaning/                        - base (no augmentation)
                meaning_with_swap/              - swap augmentation (skip identical)
                meaning_with_back_translation/  - swap + back-translation
            fold_1/                             - seed 43, stratified split
                ...

Corpus: base (1355) + identical (359, score=100) + unrelated (359, score=0), deduplicated.
Each fold is stratified on source type (original/identical/unrelated).
Swap skips identical pairs. Back-translation applied on train split only per fold
to prevent data leakage (slower but correct).
"""
from __future__ import annotations

import argparse
import os

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

FOLD_SEEDS: list[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def _translate_batch(
    batch: list[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: torch.device,
) -> list[str]:
    """Translate a single batch of texts."""
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    tokens = model.generate(**inputs, max_length=512)
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)


def back_translate_batch(
    texts: list[str],
    en_fr: tuple[MarianTokenizer, MarianMTModel],
    fr_en: tuple[MarianTokenizer, MarianMTModel],
    device: torch.device,
    batch_size: int = 32,
) -> list[str]:
    """Back-translate a list of English texts via French (EN -> FR -> EN).

    Args:
        texts: English sentences to paraphrase.
        en_fr: (tokenizer, model) for EN->FR.
        fr_en: (tokenizer, model) for FR->EN.
        device: Torch device for inference.
        batch_size: Number of sentences per batch.

    Returns:
        Back-translated English sentences.
    """
    en_fr_tokenizer, en_fr_model = en_fr
    fr_en_tokenizer, fr_en_model = fr_en

    results: list[str] = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="    Back-translating"):
        batch = texts[i : i + batch_size]
        fr_texts = _translate_batch(batch, en_fr_tokenizer, en_fr_model, device)
        en_texts = _translate_batch(fr_texts, fr_en_tokenizer, fr_en_model, device)
        results.extend(en_texts)

    return results


def back_translate_dataset(
    dataset: Dataset,
    en_fr: tuple[MarianTokenizer, MarianMTModel],
    fr_en: tuple[MarianTokenizer, MarianMTModel],
    device: torch.device,
    batch_size: int = 32,
    exclude_fingerprints: set[str] | None = None,
) -> Dataset:
    """Add back-translated pairs to a dataset.

    For each example (A, B, label), creates two new pairs:
    (bt_A, B, label) and (A, bt_B, label). Tags new pairs with source='back_translated'.
    Pairs whose (original, simplification) fingerprint appears in *exclude_fingerprints*
    are dropped to prevent data leakage into dev/test.

    Args:
        dataset: HF Dataset with columns 'original', 'simplification', 'label'.
        en_fr: (tokenizer, model) for EN->FR.
        fr_en: (tokenizer, model) for FR->EN.
        device: Torch device for inference.
        batch_size: Number of sentences per batch.
        exclude_fingerprints: Set of "original|||simplification" strings to exclude.

    Returns:
        Concatenated dataset: original rows + back-translated rows (deduplicated).
    """
    originals = list(dataset["original"])
    simplifications = list(dataset["simplification"])
    labels = list(dataset["label"])

    print(f"  Back-translating {len(originals)} originals...")
    bt_originals = back_translate_batch(originals, en_fr, fr_en, device, batch_size)

    print(f"  Back-translating {len(simplifications)} simplifications...")
    bt_simplifications = back_translate_batch(simplifications, en_fr, fr_en, device, batch_size)

    # Build augmented pairs: (bt_orig, simp) and (orig, bt_simp)
    aug_orig = bt_originals + originals
    aug_simp = simplifications + bt_simplifications
    aug_labels = labels + labels

    # Filter out pairs that collide with dev/test fingerprints
    if exclude_fingerprints:
        keep_orig, keep_simp, keep_labels = [], [], []
        n_dropped = 0
        for o, s, l in zip(aug_orig, aug_simp, aug_labels):
            if f"{o}|||{s}" in exclude_fingerprints:
                n_dropped += 1
            else:
                keep_orig.append(o)
                keep_simp.append(s)
                keep_labels.append(l)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} bt pairs colliding with dev/test")
        aug_orig, aug_simp, aug_labels = keep_orig, keep_simp, keep_labels

    augmented = Dataset.from_dict({
        "original": aug_orig,
        "simplification": aug_simp,
        "label": aug_labels,
        "source": ["back_translated"] * len(aug_orig),
    })

    # Ensure original rows are tagged
    if "source" not in dataset.column_names:
        dataset = dataset.add_column("source", ["original"] * len(dataset))

    return concatenate_datasets([dataset, augmented])


def apply_commutative_swap(dataset: Dataset) -> Dataset:
    """Apply commutative property: Meaning(a, b) = Meaning(b, a).

    Skips identical pairs (swapping identical sentences is a no-op).

    Args:
        dataset: HF Dataset with columns 'original', 'simplification', 'label'
                 and optionally 'source'.

    Returns:
        Concatenated dataset: original rows + swapped rows.
    """
    originals = list(dataset["original"])
    simplifications = list(dataset["simplification"])
    labels = list(dataset["label"])
    sources = list(dataset["source"]) if "source" in dataset.column_names else ["original"] * len(originals)

    # Ensure source column exists
    if "source" not in dataset.column_names:
        dataset = dataset.add_column("source", sources)

    # Only swap non-identical pairs
    swap_orig: list[str] = []
    swap_simp: list[str] = []
    swap_labels: list[float] = []
    for orig, simp, label, src in zip(originals, simplifications, labels, sources):
        if src == "identical":
            continue
        swap_orig.append(simp)
        swap_simp.append(orig)
        swap_labels.append(label)

    swapped = Dataset.from_dict({
        "original": swap_orig,
        "simplification": swap_simp,
        "label": swap_labels,
        "source": ["swapped"] * len(swap_orig),
    })

    return concatenate_datasets([dataset, swapped])


def create_fold_splits(
    full_dataset: Dataset,
    seed: int,
    stratify_column: str = "source",
    dev_ratio: float = 0.1,
    test_ratio: float = 0.3,
) -> DatasetDict:
    """Split a dataset into train/dev/test with stratification.

    Args:
        full_dataset: Complete dataset to split.
        seed: Random seed for reproducibility.
        stratify_column: Column name to stratify on.
        dev_ratio: Proportion of data for dev set (relative to total).
        test_ratio: Proportion of data for test set.

    Returns:
        DatasetDict with 'train', 'dev', 'test' splits.
    """
    indices = list(range(len(full_dataset)))
    strata = full_dataset[stratify_column]

    train_dev_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed, stratify=strata,
    )
    relative_dev_ratio = dev_ratio / (1 - test_ratio)
    train_dev_strata = [strata[i] for i in train_dev_idx]
    train_idx, dev_idx = train_test_split(
        train_dev_idx, test_size=relative_dev_ratio, random_state=seed, stratify=train_dev_strata,
    )

    return DatasetDict({
        "train": full_dataset.select(train_idx),
        "dev": full_dataset.select(dev_idx),
        "test": full_dataset.select(test_idx),
    })


def main() -> None:
    """Pre-generate all CSMD dataset variants (base, swap, back-translation) across k folds."""
    parser = argparse.ArgumentParser(description="Pre-generate all CSMD dataset variants to disk.")
    parser.add_argument("--output_dir", type=str, default="./data", help="Root directory for all dataset variants.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for back-translation.")
    parser.add_argument("--skip_back_translation", action="store_true", help="Skip back-translation.")
    parser.add_argument("--num_folds", type=int, default=10, help="Number of k-fold splits to generate.")
    args = parser.parse_args()

    output_dir: str = args.output_dir
    num_folds: int = args.num_folds
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Load and merge all data sources into one corpus with source tags
    print("=== Downloading datasets ===")
    meaning_raw = load_dataset("davebulaval/CSMD", "meaning")
    meaning_pool = concatenate_datasets([meaning_raw["train"], meaning_raw["dev"], meaning_raw["test"]])
    meaning_pool = meaning_pool.add_column("source", ["original"] * len(meaning_pool))

    holdout_identical = load_dataset("davebulaval/CSMD", "meaning_holdout_identical")["test"]
    holdout_identical = holdout_identical.add_column("source", ["identical"] * len(holdout_identical))

    holdout_unrelated = load_dataset("davebulaval/CSMD", "meaning_holdout_unrelated")["test"]
    holdout_unrelated = holdout_unrelated.add_column("source", ["unrelated"] * len(holdout_unrelated))

    full_dataset = concatenate_datasets([meaning_pool, holdout_identical, holdout_unrelated])

    # Deduplicate on (original, simplification) to prevent cross-split leakage
    seen: set[str] = set()
    unique_indices: list[int] = []
    for i, (o, s) in enumerate(zip(full_dataset["original"], full_dataset["simplification"])):
        key = f"{o}|||{s}"
        if key not in seen:
            seen.add(key)
            unique_indices.append(i)
    n_dupes = len(full_dataset) - len(unique_indices)
    if n_dupes > 0:
        print(f"  Removed {n_dupes} duplicate (original, simplification) pairs")
    full_dataset = full_dataset.select(unique_indices)

    print(f"  Full corpus: {len(full_dataset)} examples (after dedup)")
    print(f"    original: {len(meaning_pool)}, identical: {len(holdout_identical)}, unrelated: {len(holdout_unrelated)}")

    # Load translation models once (reused across folds)
    en_fr = None
    fr_en = None
    if not args.skip_back_translation:
        print("\n=== Loading translation models ===")
        en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr", torch_dtype=torch.bfloat16).to(device)
        fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en", torch_dtype=torch.bfloat16).to(device)
        en_fr = (en_fr_tokenizer, en_fr_model)
        fr_en = (fr_en_tokenizer, fr_en_model)
    else:
        print("\n=== Skipping back-translation ===")

    # Generate k-fold splits (stratified on source type)
    print(f"\n=== Generating {num_folds} stratified folds ===")
    fold_seeds = FOLD_SEEDS[:num_folds]

    for fold_idx, seed in enumerate(tqdm(fold_seeds, desc="Generating folds")):
        fold_dir = os.path.join(output_dir, "folds", f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # 1. Base (no augmentation) - stratified split
        base_splits = create_fold_splits(full_dataset, seed, stratify_column="source")
        base_path = os.path.join(fold_dir, "meaning")
        base_splits.save_to_disk(base_path)

        # 2. Swap (commutative property) - applied only on train
        swap_splits = DatasetDict({
            "train": apply_commutative_swap(base_splits["train"]),
            "dev": base_splits["dev"],
            "test": base_splits["test"],
        })
        swap_path = os.path.join(fold_dir, "meaning_with_swap")
        swap_splits.save_to_disk(swap_path)

        # 3. Back-translation on TRAIN ONLY, then swap. Dev/test stay clean.
        if en_fr is not None and fr_en is not None:
            # Build fingerprints of dev+test to exclude colliding bt pairs
            dev_test_fps: set[str] = set()
            for split in ["dev", "test"]:
                for o, s in zip(base_splits[split]["original"], base_splits[split]["simplification"]):
                    dev_test_fps.add(f"{o}|||{s}")

            # Also add reversed fingerprints for swap dedup
            for o, s in zip(base_splits["dev"]["original"], base_splits["dev"]["simplification"]):
                dev_test_fps.add(f"{s}|||{o}")
            for o, s in zip(base_splits["test"]["original"], base_splits["test"]["simplification"]):
                dev_test_fps.add(f"{s}|||{o}")

            print(f"\n  Back-translating fold {fold_idx} train ({len(base_splits['train'])} examples)...")
            bt_train = back_translate_dataset(
                base_splits["train"], en_fr, fr_en, device, args.batch_size,
                exclude_fingerprints=dev_test_fps,
            )
            bt_train_swapped = apply_commutative_swap(bt_train)

            # Final dedup of swapped bt pairs against dev/test
            keep_indices = [
                i for i, (o, s) in enumerate(
                    zip(bt_train_swapped["original"], bt_train_swapped["simplification"])
                )
                if f"{o}|||{s}" not in dev_test_fps
            ]
            if len(keep_indices) < len(bt_train_swapped):
                n_dropped = len(bt_train_swapped) - len(keep_indices)
                print(f"  Dropped {n_dropped} swapped pairs colliding with dev/test")
                bt_train_swapped = bt_train_swapped.select(keep_indices)
            bt_splits = DatasetDict({
                "train": bt_train_swapped,
                "dev": base_splits["dev"],
                "test": base_splits["test"],
            })
            bt_path = os.path.join(fold_dir, "meaning_with_back_translation")
            bt_splits.save_to_disk(bt_path)

        # Stats
        source_counts: dict[str, int] = {}
        for src in base_splits["train"]["source"]:
            source_counts[src] = source_counts.get(src, 0) + 1
        print(f"  Fold {fold_idx} (seed={seed}): "
              f"train={len(base_splits['train'])} {source_counts}, "
              f"dev={len(base_splits['dev'])}, "
              f"test={len(base_splits['test'])}")

    # Free GPU memory
    if en_fr is not None:
        del en_fr, fr_en
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Done ===")
    print(f"All datasets saved to {output_dir}/")
    print(f"Structure:")
    print(f"  folds/ ({num_folds} folds)")
    for fold_idx in range(num_folds):
        fold_dir = os.path.join(output_dir, "folds", f"fold_{fold_idx}")
        variants = [d for d in sorted(os.listdir(fold_dir)) if os.path.isdir(os.path.join(fold_dir, d))]
        print(f"    fold_{fold_idx}/ -> {', '.join(variants)}")


if __name__ == "__main__":
    main()
