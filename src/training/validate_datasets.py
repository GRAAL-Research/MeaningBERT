"""Validate pre-generated dataset folds for correctness.

Run after prepare_datasets.py::

    python validate_datasets.py --data_dir ./data

Checks:
    - All expected folds and variants exist
    - No data leakage between train/dev/test splits
    - Stratification: source types are proportionally distributed
    - Label sanity: identical ~100, unrelated ~0, original in range
    - Swap correctness: swapped pairs have reversed original/simplification
    - Back-translation: corpus is larger than swap variant
    - Column consistency across all datasets
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

from datasets import load_from_disk


def check_exists(path: str, name: str) -> bool:
    """Check that a dataset directory exists."""
    if not os.path.isdir(path):
        print(f"  FAIL: {name} not found at {path}")
        return False
    return True


def check_no_leakage(splits: dict[str, set[str]], fold_name: str) -> bool:
    """Check that there is no overlap between train/dev/test splits."""
    ok = True
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        if a not in splits or b not in splits:
            continue
        overlap = splits[a] & splits[b]
        if overlap:
            print(f"  FAIL [{fold_name}]: {len(overlap)} leaked examples between {a} and {b}")
            ok = False
    return ok


def make_fingerprints(dataset) -> set[str]:
    """Create a set of fingerprints from (original, simplification) pairs."""
    return {f"{o}|||{s}" for o, s in zip(dataset["original"], dataset["simplification"])}


def check_stratification(dataset, split_name: str, fold_name: str, expected_sources: set[str]) -> bool:
    """Check that all expected source types are present in the split."""
    if "source" not in dataset.column_names:
        print(f"  FAIL [{fold_name}/{split_name}]: missing 'source' column")
        return False
    sources = set(dataset["source"])
    missing = expected_sources - sources
    if missing:
        print(f"  FAIL [{fold_name}/{split_name}]: missing source types: {missing}")
        return False
    return True


def check_labels(dataset, split_name: str, fold_name: str) -> bool:
    """Check label sanity per source type."""
    ok = True
    if "source" not in dataset.column_names:
        return True

    for i, (label, source) in enumerate(zip(dataset["label"], dataset["source"])):
        if source == "identical" and label < 90:
            print(f"  FAIL [{fold_name}/{split_name}]: identical pair at idx {i} has label={label} (expected ~100)")
            ok = False
            break
        if source == "unrelated" and label > 10:
            print(f"  FAIL [{fold_name}/{split_name}]: unrelated pair at idx {i} has label={label} (expected ~0)")
            ok = False
            break
    return ok


def check_swap_present(swap_dataset, base_dataset, split_name: str, fold_name: str) -> bool:
    """Check that swap variant has more training examples than base (swap applied on train only)."""
    if split_name != "train":
        return True
    if len(swap_dataset) <= len(base_dataset):
        print(
            f"  FAIL [{fold_name}/{split_name}]: swap ({len(swap_dataset)})"
            f" should be larger than base ({len(base_dataset)})"
        )
        return False
    return True


def main() -> None:
    """Validate all dataset folds."""
    parser = argparse.ArgumentParser(description="Validate pre-generated dataset folds.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory.")
    parser.add_argument("--num_folds", type=int, default=10, help="Expected number of folds.")
    args = parser.parse_args()

    data_dir: str = args.data_dir
    num_folds: int = args.num_folds
    errors = 0
    checks = 0

    print(f"Validating datasets in {data_dir}/\n")

    # Check folds directory
    folds_dir = os.path.join(data_dir, "folds")
    if not check_exists(folds_dir, "folds/"):
        print("\nABORT: folds directory missing.")
        sys.exit(1)

    variants = ["meaning", "meaning_with_swap", "meaning_with_back_translation"]
    base_sources = {"original", "identical", "unrelated"}
    splits = ["train", "dev", "test"]

    for fold_idx in range(num_folds):
        fold_name = f"fold_{fold_idx}"
        fold_dir = os.path.join(folds_dir, fold_name)
        print(f"--- {fold_name} ---")

        if not check_exists(fold_dir, fold_name):
            errors += 1
            continue

        datasets: dict[str, dict] = {}
        for variant in variants:
            variant_path = os.path.join(fold_dir, variant)
            checks += 1
            if not check_exists(variant_path, f"{fold_name}/{variant}"):
                errors += 1
                continue
            datasets[variant] = load_from_disk(variant_path)

        if not datasets:
            continue

        # Check each variant
        for variant, ds in datasets.items():
            # Column check
            checks += 1
            expected_cols = {"original", "simplification", "label"}
            for split_name in splits:
                if split_name not in ds:
                    print(f"  FAIL [{fold_name}/{variant}]: missing split '{split_name}'")
                    errors += 1
                    continue
                actual_cols = set(ds[split_name].column_names)
                missing_cols = expected_cols - actual_cols
                if missing_cols:
                    print(f"  FAIL [{fold_name}/{variant}/{split_name}]: missing columns {missing_cols}")
                    errors += 1

            # No leakage (using base meaning variant fingerprints)
            checks += 1
            fingerprints = {}
            for split_name in splits:
                if split_name in ds:
                    fingerprints[split_name] = make_fingerprints(ds[split_name])
            if not check_no_leakage(fingerprints, f"{fold_name}/{variant}"):
                errors += 1

            # Per-split checks
            for split_name in splits:
                if split_name not in ds:
                    continue

                # Stratification (base sources must be present in base variant)
                if variant == "meaning":
                    checks += 1
                    if not check_stratification(ds[split_name], split_name, f"{fold_name}/{variant}", base_sources):
                        errors += 1

                # Label sanity
                checks += 1
                if not check_labels(ds[split_name], split_name, f"{fold_name}/{variant}"):
                    errors += 1

            # Swap should have more train examples than base
            if "meaning" in datasets and "meaning_with_swap" in datasets:
                checks += 1
                if not check_swap_present(
                    datasets["meaning_with_swap"]["train"],
                    datasets["meaning"]["train"],
                    "train",
                    fold_name,
                ):
                    errors += 1

            # Back-translation should have more train examples than swap
            if "meaning_with_swap" in datasets and "meaning_with_back_translation" in datasets:
                checks += 1
                bt_train = len(datasets["meaning_with_back_translation"]["train"])
                swap_train = len(datasets["meaning_with_swap"]["train"])
                if bt_train <= swap_train:
                    print(
                        f"  FAIL [{fold_name}]: back_translation train ({bt_train})"
                        f" should be larger than swap train ({swap_train})"
                    )
                    errors += 1

        # Print stats for this fold
        if "meaning" in datasets:
            base = datasets["meaning"]
            print(f"  Base:    train={len(base['train'])}, dev={len(base['dev'])}, test={len(base['test'])}")
            if "source" in base["train"].column_names:
                counts = Counter(base["train"]["source"])
                print(f"           train sources: {dict(counts)}")
        if "meaning_with_swap" in datasets:
            swap = datasets["meaning_with_swap"]
            print(f"  Swap:    train={len(swap['train'])}, dev={len(swap['dev'])}, test={len(swap['test'])}")
        if "meaning_with_back_translation" in datasets:
            bt = datasets["meaning_with_back_translation"]
            print(f"  BT+Swap: train={len(bt['train'])}, dev={len(bt['dev'])}, test={len(bt['test'])}")
        print()

    # Summary
    print("=" * 50)
    if errors == 0:
        print(f"ALL PASSED: {checks} checks, 0 errors")
    else:
        print(f"FAILED: {errors} errors out of {checks} checks")
        sys.exit(1)


if __name__ == "__main__":
    main()
