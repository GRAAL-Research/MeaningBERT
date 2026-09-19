"""Split the merged corpus without leaking a source sentence across splits.

CSMD v1 splits at the row level. Since a source sentence carries several simplifications,
that puts the same sentence on both sides of the wall: measured on the published splits,
**82.8 percent of test rows have their source sentence in train**, and 76.3 percent of the
identical and unrelated "holdout" pairs do too. ``validate_datasets.py`` reported no
leakage because it compares exact ``(original, simplification)`` pairs and never looks at
the source sentence. See ``docs/H5-fuite-par-phrase-source.md``.

The corpora arriving in v2 make this worse, not better: SimpEval carries roughly 2400
simplifications over about 60 source sentences. A row-level split of that is not a split.

So the unit of splitting here is the **source sentence group**, never the row.

The module also carves a sanity holdout. The v1 identical and unrelated pairs are used
both as training signal and as the sanity check, which is why
``docs/H1-diagnostic-calibration.md`` finds the checks measuring something in-distribution.
Here a fraction of groups is reserved: their sanity pairs go to the ``sanity`` split, their
ordinary pairs go to ``test``, and the whole group stays out of training.
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

from datasets import Dataset, DatasetDict

from data.harmonize import normalise_text, pair_key

SANITY_SOURCES: frozenset[str] = frozenset({"identical", "unrelated"})


def group_key(original: str) -> str:
    """The unit of splitting: the normalised source sentence."""
    return normalise_text(original)


@dataclass
class SplitReport:
    """What a split actually produced, and whether it holds."""

    rows: dict[str, int] = field(default_factory=dict)
    groups: dict[str, int] = field(default_factory=dict)
    sources: dict[str, dict[str, int]] = field(default_factory=dict)
    corpora: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped_duplicate_pairs: int = 0

    def summary(self) -> str:
        """Human-readable table of the split."""
        lines = [f"dropped exact duplicate pairs: {self.dropped_duplicate_pairs}", ""]
        lines.append(f"{'split':10} {'rows':>7} {'groups':>8}  source tags")
        lines.append("-" * 70)
        for split in self.rows:
            tags = ", ".join(f"{k}={v}" for k, v in sorted(self.sources.get(split, {}).items()))
            lines.append(f"{split:10} {self.rows[split]:>7} {self.groups[split]:>8}  {tags}")
        return "\n".join(lines)


class LeakageError(AssertionError):
    """Raised when a split shares a source sentence or an exact pair across sides."""


def _dedupe_exact_pairs(dataset: Dataset) -> tuple[Dataset, int]:
    """Drop repeated ``(original, simplification)`` pairs, keeping the first occurrence.

    CSMD v1 ships 12 pairs present in both train and test. Splitting by group would keep
    both copies on the same side, but the duplicate itself is still spurious weight on one
    example, so it goes.
    """
    seen: set[str] = set()
    keep: list[int] = []
    for index, (original, simplification) in enumerate(zip(dataset["original"], dataset["simplification"])):
        key = pair_key(original, simplification)
        if key not in seen:
            seen.add(key)
            keep.append(index)
    return dataset.select(keep), len(dataset) - len(keep)


def _group_rows(dataset: Dataset) -> dict[str, list[int]]:
    """Map each source sentence to the row indices that share it."""
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for index, original in enumerate(dataset["original"]):
        groups[group_key(original)].append(index)
    return dict(groups)


def _assign_greedy(
    group_names: list[int] | list[str],
    sizes: dict,
    targets: dict[str, float],
) -> dict[str, list]:
    """Assign groups to splits, filling whichever split is furthest below its target.

    Groups have very different sizes, from a single row to a couple of thousand once
    SimpEval arrives, so proportional sampling of groups misses the row-count targets
    badly. Largest-first placement keeps the error bounded by the largest group.
    """
    ordered = sorted(group_names, key=lambda name: -sizes[name])
    total = sum(sizes.values())
    assigned: dict[str, list] = {split: [] for split in targets}
    placed: dict[str, int] = {split: 0 for split in targets}

    for name in ordered:
        deficits = {split: targets[split] * total - placed[split] for split in targets}
        best = max(deficits, key=lambda split: deficits[split])
        assigned[best].append(name)
        placed[best] += sizes[name]
    return assigned


def _tally(dataset: Dataset, column: str) -> dict[str, int]:
    return dict(collections.Counter(dataset[column]))


def split_by_source_sentence(
    dataset: Dataset,
    seed: int = 42,
    dev_frac: float = 0.10,
    test_frac: float = 0.20,
    sanity_frac: float = 0.30,
) -> tuple[DatasetDict, SplitReport]:
    """Split *dataset* so that no source sentence appears in two splits.

    Args:
        dataset: Merged, harmonised corpus.
        seed: Shuffling seed.
        dev_frac: Share of non-reserved rows for dev.
        test_frac: Share of non-reserved rows for test.
        sanity_frac: Share of sanity-carrying groups reserved as a true holdout. Their
            identical and unrelated rows land in ``sanity``, their ordinary rows in
            ``test``, and none of them reach ``train``.

    Returns:
        A dict with ``train``, ``dev``, ``test`` and ``sanity``, plus the report.

    Raises:
        ValueError: If the requested fractions do not leave room for a training set.
    """
    if dev_frac + test_frac >= 1.0:
        raise ValueError(f"dev_frac + test_frac must leave room for train, got {dev_frac + test_frac}")
    if not 0.0 <= sanity_frac <= 1.0:
        raise ValueError(f"sanity_frac must be in [0, 1], got {sanity_frac}")

    deduped, dropped = _dedupe_exact_pairs(dataset)
    groups = _group_rows(deduped)
    sources = deduped["source"]

    sanity_groups = sorted(
        name for name, indices in groups.items() if any(sources[index] in SANITY_SOURCES for index in indices)
    )
    plain_groups = sorted(set(groups) - set(sanity_groups))

    rng = random.Random(seed)
    rng.shuffle(sanity_groups)
    n_reserved = int(round(len(sanity_groups) * sanity_frac))
    reserved, sanity_in_play = set(sanity_groups[:n_reserved]), sanity_groups[n_reserved:]

    assignable = plain_groups + sanity_in_play
    rng.shuffle(assignable)
    sizes = {name: len(groups[name]) for name in assignable}
    targets = {"train": 1.0 - dev_frac - test_frac, "dev": dev_frac, "test": test_frac}
    assigned = _assign_greedy(assignable, sizes, targets) if assignable else {k: [] for k in targets}

    buckets: dict[str, list[int]] = {"train": [], "dev": [], "test": [], "sanity": []}
    for split, names in assigned.items():
        for name in names:
            buckets[split].extend(groups[name])
    for name in reserved:
        for index in groups[name]:
            buckets["sanity" if sources[index] in SANITY_SOURCES else "test"].append(index)

    splits = DatasetDict({name: deduped.select(sorted(indices)) for name, indices in buckets.items()})

    report = SplitReport(dropped_duplicate_pairs=dropped)
    for name, subset in splits.items():
        report.rows[name] = len(subset)
        report.groups[name] = len({group_key(o) for o in subset["original"]}) if len(subset) else 0
        report.sources[name] = _tally(subset, "source") if len(subset) else {}
        report.corpora[name] = _tally(subset, "corpus") if len(subset) else {}
    return splits, report


#: Pairs of splits allowed to share a source sentence.
#:
#: ``test`` and ``sanity`` deliberately do. A reserved group contributes its identical and
#: unrelated rows to ``sanity`` and its ordinary rows to ``test``; the point of reserving
#: it is that it never reaches ``train``. Sharing a source sentence between two evaluation
#: sets does not invalidate either measurement, it only means the two reported numbers are
#: not statistically independent, which nobody claims they are. Every other pair, and in
#: particular anything involving ``train``, must be disjoint.
DEFAULT_ALLOWED_GROUP_OVERLAPS: frozenset[frozenset[str]] = frozenset({frozenset({"test", "sanity"})})


def assert_no_leakage(
    splits: DatasetDict,
    train_only: Optional[Iterable[str]] = None,
    allowed_group_overlaps: frozenset[frozenset[str]] = DEFAULT_ALLOWED_GROUP_OVERLAPS,
) -> None:
    """Fail loudly if any source sentence or exact pair spans two splits.

    Checks the source sentence, not just the exact pair. Checking only the pair is what let
    CSMD v1 ship with 82.8 percent of its test rows sharing a source sentence with train.

    Args:
        splits: Output of :func:`split_by_source_sentence`.
        train_only: Splits that must never intersect ``train``. Defaults to every other
            split.
        allowed_group_overlaps: Split pairs permitted to share a source sentence. Exact
            pair duplication is never permitted, whatever this contains.

    Raises:
        LeakageError: Listing every overlap found, not just the first.
    """
    names = list(splits)
    keys = {name: {group_key(o) for o in splits[name]["original"]} for name in names}
    pairs = {
        name: {pair_key(o, s) for o, s in zip(splits[name]["original"], splits[name]["simplification"])}
        for name in names
    }

    problems: list[str] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared_groups = keys[left] & keys[right]
            if shared_groups and frozenset({left, right}) not in allowed_group_overlaps:
                sample = sorted(shared_groups)[:2]
                problems.append(
                    f"{left} and {right} share {len(shared_groups)} source sentence(s), e.g. {sample}"
                )
            shared_pairs = pairs[left] & pairs[right]
            if shared_pairs:
                problems.append(f"{left} and {right} share {len(shared_pairs)} exact pair(s)")

    for name in train_only or [n for n in names if n != "train"]:
        if name in keys and "train" in keys and keys[name] & keys["train"]:
            problems.append(f"{name} must be disjoint from train but shares {len(keys[name] & keys['train'])} groups")

    if problems:
        raise LeakageError("leakage detected:\n  - " + "\n  - ".join(sorted(set(problems))))
