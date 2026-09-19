"""Measure source-sentence leakage in the corpus the sweep actually trained on.

``src/training/validate_datasets.py`` reports no leakage, and it is right about what it
checks: it compares exact ``(original, simplification)`` fingerprints. It never looks at
the source sentence.

That matters because a source sentence carries several simplifications. CSMD holds 2073
rows over 493 distinct source sentences, so a row-level split scatters the same sentence
across train, dev and test. The model then meets the test sentence during training under a
different simplification.

This script reproduces the real folds. ``create_fold_splits`` is deterministic given its
seed, so re-running it on the same merged corpus reconstructs exactly what the sweep saw,
without needing the fold directories.

Run::

    PYTHONPATH=src python src/diagnostics/leakage_audit.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field

from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from data.harmonize import normalise_text, pair_key

FOLD_SEEDS: list[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


@dataclass
class FoldLeakage:
    """Leakage measured on one reproduced fold."""

    seed: int
    n_train: int
    n_dev: int
    n_test: int
    test_rows_leaked: int
    dev_rows_leaked: int
    test_groups_leaked: int
    n_test_groups: int
    exact_pairs_train_test: int

    @property
    def test_row_leak_pct(self) -> float:
        return 100.0 * self.test_rows_leaked / max(1, self.n_test)

    @property
    def dev_row_leak_pct(self) -> float:
        return 100.0 * self.dev_rows_leaked / max(1, self.n_dev)

    @property
    def test_group_leak_pct(self) -> float:
        return 100.0 * self.test_groups_leaked / max(1, self.n_test_groups)


@dataclass
class LeakageReport:
    """Leakage across every reproduced fold."""

    n_rows: int
    n_groups: int
    folds: list[FoldLeakage] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable verdict."""
        rows = [f.test_row_leak_pct for f in self.folds]
        groups = [f.test_group_leak_pct for f in self.folds]
        dev = [f.dev_row_leak_pct for f in self.folds]
        exact = [f.exact_pairs_train_test for f in self.folds]
        lines = [
            f"Merged corpus: {self.n_rows} rows over {self.n_groups} distinct source sentences "
            f"({self.n_rows / max(1, self.n_groups):.1f} simplifications per sentence).",
            "",
            f"{'seed':>5} {'train':>7} {'dev':>6} {'test':>7} {'test rows leaked':>18} {'test groups leaked':>20} {'exact dupes':>12}",
            "-" * 82,
        ]
        for fold in self.folds:
            lines.append(
                f"{fold.seed:>5} {fold.n_train:>7} {fold.n_dev:>6} {fold.n_test:>7} "
                f"{fold.test_rows_leaked:>7} ({fold.test_row_leak_pct:5.1f}%) "
                f"{fold.test_groups_leaked:>9} ({fold.test_group_leak_pct:5.1f}%) "
                f"{fold.exact_pairs_train_test:>12}"
            )
        lines += [
            "-" * 82,
            f"median test rows leaked:   {statistics.median(rows):.1f}%",
            f"median test groups leaked: {statistics.median(groups):.1f}%",
            f"median dev rows leaked:    {statistics.median(dev):.1f}%",
            f"median exact duplicate pairs train/test: {statistics.median(exact):.0f}",
            "",
            "'Leaked' means the row's source sentence is also present in train, so the model",
            "met that sentence during training under a different simplification. The exact-pair",
            "check that validate_datasets.py performs would report zero for most of these.",
        ]
        return "\n".join(lines)


def build_merged_corpus() -> Dataset:
    """Rebuild the merged corpus exactly as ``prepare_datasets.main`` does."""
    meaning = load_dataset("davebulaval/CSMD", "meaning")
    pool = concatenate_datasets([meaning["train"], meaning["dev"], meaning["test"]])
    pool = pool.add_column("source", ["original"] * len(pool))

    identical = load_dataset("davebulaval/CSMD", "meaning_holdout_identical")["test"]
    identical = identical.add_column("source", ["identical"] * len(identical))
    unrelated = load_dataset("davebulaval/CSMD", "meaning_holdout_unrelated")["test"]
    unrelated = unrelated.add_column("source", ["unrelated"] * len(unrelated))

    full = concatenate_datasets([pool, identical, unrelated])

    seen: set[str] = set()
    keep: list[int] = []
    for index, (original, simplification) in enumerate(zip(full["original"], full["simplification"])):
        key = f"{original}|||{simplification}"
        if key not in seen:
            seen.add(key)
            keep.append(index)
    return full.select(keep)


def reproduce_fold(full: Dataset, seed: int, dev_ratio: float = 0.1, test_ratio: float = 0.3) -> FoldLeakage:
    """Re-run ``create_fold_splits`` for *seed* and measure the leakage it produces."""
    indices = list(range(len(full)))
    strata = full["source"]

    train_dev_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=strata)
    relative_dev = dev_ratio / (1 - test_ratio)
    train_idx, dev_idx = train_test_split(
        train_dev_idx,
        test_size=relative_dev,
        random_state=seed,
        stratify=[strata[i] for i in train_dev_idx],
    )

    originals = full["original"]
    simplifications = full["simplification"]

    def groups(idx: list[int]) -> set[str]:
        return {normalise_text(originals[i]) for i in idx}

    def pairs(idx: list[int]) -> set[str]:
        return {pair_key(originals[i], simplifications[i]) for i in idx}

    train_groups = groups(train_idx)
    test_groups = groups(test_idx)

    return FoldLeakage(
        seed=seed,
        n_train=len(train_idx),
        n_dev=len(dev_idx),
        n_test=len(test_idx),
        test_rows_leaked=sum(1 for i in test_idx if normalise_text(originals[i]) in train_groups),
        dev_rows_leaked=sum(1 for i in dev_idx if normalise_text(originals[i]) in train_groups),
        test_groups_leaked=len(test_groups & train_groups),
        n_test_groups=len(test_groups),
        exact_pairs_train_test=len(pairs(train_idx) & pairs(test_idx)),
    )


def audit(seeds: list[int]) -> LeakageReport:
    """Reproduce every fold and measure leakage."""
    full = build_merged_corpus()
    report = LeakageReport(
        n_rows=len(full),
        n_groups=len({normalise_text(o) for o in full["original"]}),
    )
    report.folds = [reproduce_fold(full, seed) for seed in seeds]
    return report


def main() -> None:
    """Audit the folds the sweep trained on for source-sentence leakage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="*", default=FOLD_SEEDS, help="Fold seeds to reproduce.")
    parser.add_argument("--json-out", default=None, help="Optional path for the full per-fold audit.")
    args = parser.parse_args()

    report = audit(args.seeds)
    print(report.summary())

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "n_rows": report.n_rows,
                    "n_groups": report.n_groups,
                    "folds": [
                        {
                            **vars(fold),
                            "test_row_leak_pct": fold.test_row_leak_pct,
                            "test_group_leak_pct": fold.test_group_leak_pct,
                            "dev_row_leak_pct": fold.dev_row_leak_pct,
                        }
                        for fold in report.folds
                    ],
                },
                handle,
                indent=2,
            )
        print(f"\nFull audit written to {args.json_out}")


if __name__ == "__main__":
    main()
