"""Merge the v3 polarity corpora into one training corpus, with the probes held out.

This is experiment 2 of ``docs/v3-echelle-signee.md``, the step between the loaders and the
polarity head. It does the one thing the contract forbids the loaders to do: reconcile
label spaces that were written by different people for different papers.

**What trains and what does not.** VitaminC and SICK train. MoNLI and NaN-NLI do not: they
are diagnostic probes of 1 202 and 258 pairs, already two of the three suites in
``src/diagnostics/dissociation.py``, and a probe that has been trained on measures nothing.
They are built and saved beside the corpus so the evaluation can reach them, never inside
it. This is enforced, not documented: :func:`build` raises if a probe corpus ever appears
in a split.

**Why VitaminC is capped.** It is 371 000 training rows against SICK's 4 439, so left whole
it would decide the head on its own and SICK would contribute noise. The cap is applied
per class so the ratio between the three classes is preserved rather than silently
rebalanced, and the drawn rows are chosen with a fixed seed so the corpus is reproducible.

Run::

    PYTHONPATH=src python src/data/build_polarity_corpus.py --out datastore/polarity
"""

from __future__ import annotations

import collections
import json
import random
from typing import Final, Optional

import click
from datasets import Dataset, DatasetDict, concatenate_datasets

from data.augment import generate_identical, generate_unrelated, swap
from data.loaders import monli, nan_nli, sick, vitaminc
from data.schema import POLARITY_CLASSES, is_symmetric_polarity
from data.splits import group_key

#: Corpora the polarity head learns from.
TRAINING_CORPORA: Final[dict] = {"vitaminc": vitaminc, "sick": sick}

#: Corpora kept entirely outside training. They are probes, and a probe that has been
#: trained on measures nothing.
PROBE_CORPORA: Final[dict] = {"monli": monli, "nan_nli": nan_nli}

#: How each native label space maps onto the three unified classes.
#:
#: ``fact3`` is the only reconciliation that needs an argument, and it is short: a claim
#: SUPPORTED by its evidence is entailed by it, a claim REFUTED by its evidence is
#: contradicted by it, and NOT ENOUGH INFO is the definition of neutral. The relation is
#: the same one under a fact-checking name. ``nli3`` is the identity, which is the point of
#: having made the loaders name their classes instead of emitting integers.
POLARITY_MAP: Final[dict[str, dict[str, str]]] = {
    "nli3": {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"},
    "fact3": {"SUPPORTS": "entailment", "NOT ENOUGH INFO": "neutral", "REFUTES": "contradiction"},
}

#: Split hints a training corpus may carry. A row with an empty hint has no home here.
SPLITS: Final[tuple[str, ...]] = ("train", "dev", "test")


#: Which augmentation feeds which class. Each augmented row's polarity is DERIVED, never
#: copied, which is the rule the whole thing rests on:
#:
#: * a mirrored contradiction is still a contradiction, because if A denies B then B denies
#:   A. Mirrored entailments and neutrals are dropped rather than kept: the relation does
#:   not survive the swap, and these corpora carry no magnitude label either, so the row
#:   would arrive with no target at all.
#: * two sentences drawn at random and kept only when they barely overlap are neutral. This
#:   is the most valuable of the three, because it teaches the distinction the whole signed
#:   scale rests on: unrelated scores 0, it does not score -100.
#: * a sentence entails itself. Weak on its own, and included for a mechanical reason: it
#:   is the only lever that grows the entailment class, and without it mirroring alone
#:   would take contradiction from 50 641 to 101 282 against 51 274 entailments, teaching
#:   the head to over-predict the one class whose sign flips the score.
#: How much more than the quota to ask the generators for, before cutting back. See
#: :func:`augment_polarity` for why asking for exactly the quota under-delivers.
_OVERSHOOT: Final[float] = 1.4

AUGMENTATION_TARGETS: Final[dict[str, str]] = {
    "swapped": "contradiction",
    "unrelated": "neutral",
    "identical": "entailment",
}


class BuildError(RuntimeError):
    """Raised when the merged corpus would be wrong rather than merely smaller."""


def unify(dataset: Dataset) -> Dataset:
    """Fill the ``polarity`` column from ``polarity_raw`` and ``polarity_scheme``.

    Args:
        dataset: A loader output, one scheme throughout.

    Returns:
        The same rows with ``polarity`` filled with the unified class index.

    Raises:
        BuildError: If the scheme is unknown, or a raw value has no mapping. Neither is a
            row to drop: both mean a label space moved, and the rest of the corpus is then
            suspect too.
    """
    schemes = set(dataset["polarity_scheme"])
    unknown = schemes - POLARITY_MAP.keys()
    if unknown:
        raise BuildError(f"no unified mapping for polarity scheme(s) {sorted(unknown)}")

    mapping = POLARITY_MAP[next(iter(schemes))]
    missing = sorted(set(dataset["polarity_raw"]) - mapping.keys())
    if missing:
        raise BuildError(f"no unified class for raw polarity value(s) {missing}")

    classes = [float(POLARITY_CLASSES[mapping[raw]]) for raw in dataset["polarity_raw"]]
    return dataset.remove_columns(["polarity"]).add_column("polarity", classes)


def cap_per_class(dataset: Dataset, cap: Optional[int], seed: int) -> Dataset:
    """Keep at most *cap* rows of each polarity class, drawn reproducibly.

    Capping per class rather than over the whole dataset preserves the corpus's own class
    ratio instead of quietly rebalancing it, which would be a second, undeclared
    intervention on top of the size one.

    Args:
        dataset: Rows with ``polarity`` already filled.
        cap: Maximum rows per class, or ``None`` to keep everything.
        seed: Draw seed, so the corpus is reproducible.

    Returns:
        The kept rows, in their original order.
    """
    if cap is None:
        return dataset

    by_class: dict[float, list[int]] = collections.defaultdict(list)
    for index, value in enumerate(dataset["polarity"]):
        by_class[value].append(index)

    rng = random.Random(seed)
    kept: list[int] = []
    for value in sorted(by_class):
        indices = by_class[value]
        kept.extend(indices if len(indices) <= cap else rng.sample(indices, cap))
    return dataset.select(sorted(kept))


def assert_no_pair_leakage(splits: dict[str, Dataset]) -> None:
    """Refuse a corpus whose evaluation pairs also appear in training.

    Args:
        splits: The built splits, keyed by name.

    Raises:
        BuildError: If any pair appears in train and also in dev or test. The corpora
            publish their own splits and those are trusted, but they were built
            independently of each other, so nothing guarantees the union is clean.
    """
    def keys(dataset: Dataset) -> set[tuple[str, str]]:
        return {
            (left.strip().lower(), right.strip().lower())
            for left, right in zip(dataset["original"], dataset["simplification"])
        }

    train = keys(splits["train"])
    for name in ("dev", "test"):
        shared = train & keys(splits[name])
        if shared:
            raise BuildError(
                f"{len(shared)} pair(s) appear in both train and {name}, e.g. {sorted(shared)[:2]}"
            )


def _tagged(before: Dataset, after: Dataset, source: str) -> Dataset:
    """Keep only the rows a generator appended, identified by their tag.

    The generators return input plus output because that is what the v2 pipeline wants.
    Here only the new rows are needed, and they are found by their ``source`` tag rather
    than by a row count, so a generator that ever reorders its output cannot corrupt this.
    """
    del before
    return after.filter(lambda row: row["source"] == source)


def augment_polarity(
    train: Dataset,
    forbidden_groups: set[str],
    per_class: int = 25_000,
    seed: int = 42,
) -> tuple[Dataset, dict[str, int]]:
    """Grow the training split with rows whose polarity can be derived.

    Train only. Augmenting an evaluation split would measure the model's ability to
    recognise our own generators rather than its grasp of the relation.

    Args:
        train: The merged training split, polarity already unified.
        forbidden_groups: Source sentences owned by dev or test. Swapping moves the
            simplification into the source position, so a mirrored row can land in a group
            that was deliberately held out; ``swap`` drops those.
        per_class: How many rows to add per class. Equal quotas, so the class balance the
            cap established is not undone by the augmentation.
        seed: Draw seed.

    Returns:
        The augmented split and a census of what each augmentation contributed.
    """
    rng = random.Random(seed)
    census: dict[str, int] = {}
    pieces: list[Dataset] = [train]
    # Ask the generators for more than the quota, then cut back to it. They under-deliver
    # on purpose: the unrelated generator rejects any draw whose two sentences overlap too
    # much, and the identical one skips a sentence it has already used. Asking for exactly
    # the quota would quietly return less than the quota, and the three classes would stop
    # growing by the same amount, which is the one thing this is here to guarantee.
    ratio = max(per_class * _OVERSHOOT / max(len(train), 1), 0.0)

    mirrored = _tagged(train, swap(train, forbidden_groups=forbidden_groups), "swapped")
    # Only the contradictions survive the swap. The rest lost their polarity to NaN inside
    # ``swap`` itself, which is where that rule lives.
    mirrored = mirrored.filter(is_symmetric_polarity)
    if len(mirrored) > per_class:
        mirrored = mirrored.select(sorted(rng.sample(range(len(mirrored)), per_class)))
    census["swapped"] = len(mirrored)
    pieces.append(mirrored)

    for source, generator in (("unrelated", generate_unrelated), ("identical", generate_identical)):
        added = _tagged(train, generator(train, ratio=ratio, seed=seed), source)
        if len(added) > per_class:
            added = added.select(sorted(rng.sample(range(len(added)), per_class)))
        census[source] = len(added)
        pieces.append(added)

    columns = train.column_names
    return concatenate_datasets([piece.select_columns(columns) for piece in pieces]), census


def augment_and_verify(splits: dict[str, Dataset], per_class: int, seed: int) -> dict[str, int]:
    """Augment the training split in place and re-check the wall it could have breached.

    Extracted from :func:`build` so it can be tested: ``build`` needs the network, and the
    check that matters most here is the one that runs AFTER new rows exist.

    Args:
        splits: The built splits, mutated in place on ``train``.
        per_class: Rows to add per class.
        seed: Draw seed.

    Returns:
        The census of what each augmentation contributed.

    Raises:
        BuildError: If augmentation put an evaluation pair into training.
    """
    # The groups the splitter deliberately held out. Swapping moves the simplification into
    # the source position, which is the one way augmentation can bridge them, and it is
    # invisible to a check that only compares whole pairs.
    forbidden = {
        group_key(sentence) for name in ("dev", "test") for sentence in splits[name]["original"]
    }
    splits["train"], census = augment_polarity(splits["train"], forbidden, per_class=per_class, seed=seed)
    # Re-checked AFTER augmentation, not only before: the rows that could leak are the ones
    # that did not exist when the first check ran.
    assert_no_pair_leakage(splits)
    return census


def build(cap: Optional[int] = 50_000, seed: int = 42, augment_per_class: int = 0) -> tuple[DatasetDict, DatasetDict, dict]:
    """Build the merged polarity corpus and the held-out probes.

    Args:
        cap: Maximum training rows per class per corpus.
        seed: Draw seed for the cap.
        augment_per_class: Rows to add per class by augmentation, 0 to add none. This is
            the ``_none`` against ``_full`` axis of the v2 campaign, where the comparison
            reversed the conclusion twice, which is why it is measured and not assumed.

    Returns:
        The corpus splits, the probes, and a census describing both.

    Raises:
        BuildError: If a probe corpus leaks into a split, or a label space cannot be
            reconciled, or an evaluation pair appears in training.
    """
    pieces: dict[str, list[Dataset]] = {name: [] for name in SPLITS}
    census: dict = {"cap_per_class": cap, "seed": seed, "corpora": {}, "probes": {}}

    for name, module in TRAINING_CORPORA.items():
        unified = unify(module.load())
        census["corpora"][name] = {}
        for split in SPLITS:
            rows = unified.filter(lambda row, s=split: row["split_hint"] == s)
            if not len(rows):
                continue
            # Only the training split is capped. Shrinking an evaluation split would make
            # the numbers cheaper to compute and harder to compare with anything.
            rows = cap_per_class(rows, cap, seed) if split == "train" else rows
            pieces[split].append(rows)
            census["corpora"][name][split] = len(rows)

    splits = {}
    for split in SPLITS:
        if not pieces[split]:
            raise BuildError(f"no corpus contributed a {split} split")
        columns = pieces[split][0].column_names
        splits[split] = concatenate_datasets([piece.select_columns(columns) for piece in pieces[split]])

    for split, dataset in splits.items():
        leaked = set(dataset["corpus"]) & PROBE_CORPORA.keys()
        if leaked:
            raise BuildError(f"probe corpus {sorted(leaked)} reached the {split} split; probes never train")

    assert_no_pair_leakage(splits)

    if augment_per_class:
        census["augmentation"] = augment_and_verify(splits, augment_per_class, seed)

    probes = {}
    for name, module in PROBE_CORPORA.items():
        probes[name] = unify(module.load())
        census["probes"][name] = len(probes[name])

    for split, dataset in splits.items():
        counts = collections.Counter(dataset["polarity"])
        census[f"{split}_classes"] = {
            label: counts.get(float(index), 0) for label, index in POLARITY_CLASSES.items()
        }
        census[f"{split}_rows"] = len(dataset)

    return DatasetDict(splits), DatasetDict(probes), census


@click.command()
@click.option("--out", required=True, help="Directory to save the corpus and the probes into.")
@click.option("--cap", default=50_000, show_default=True, help="Max training rows per class per corpus; 0 for none.")
@click.option("--seed", default=42, show_default=True)
@click.option(
    "--augment-per-class",
    default=0,
    show_default=True,
    help="Rows added per class by augmentation; 0 builds the _none condition.",
)
def main(out: str, cap: int, seed: int, augment_per_class: int) -> None:
    """Build, report and save."""
    corpus, probes, census = build(cap=cap or None, seed=seed, augment_per_class=augment_per_class)
    corpus.save_to_disk(f"{out}/corpus")
    probes.save_to_disk(f"{out}/probes")
    with open(f"{out}/census.json", "w", encoding="utf-8") as handle:
        json.dump(census, handle, indent=2, ensure_ascii=False)

    for split in SPLITS:
        click.echo(f"{split:6} {census[f'{split}_rows']:>7,}  {census[f'{split}_classes']}")
    if "augmentation" in census:
        click.echo(f"augmentation : {census['augmentation']}")
    click.echo(f"sondes tenues a l'ecart : {census['probes']}")
    click.echo(f"corpus : {out}/corpus   sondes : {out}/probes   recensement : {out}/census.json")


if __name__ == "__main__":
    main()
