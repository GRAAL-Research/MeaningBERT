"""Training-set augmentation, applied after splitting and to the train split only.

Three augmentations, **applied together or not at all**:

* **swap**, which encodes the commutative property of the meaning function,
  ``Meaning(A, B) = Meaning(B, A)``;
* **back-translation** through French, which paraphrases one side of a pair while its
  label stays put. The pivot language is a paraphrasing detour; not one non-English
  sentence enters the corpus.
* **generated sanity pairs**, ``(A, A)`` scored 100 and ``(A, B)`` scored 0 for an
  unrelated ``B``. CSMD ships only 359 of each, and the identical sanity check fails on
  100 percent of the sweep's runs; generating more attacks that directly.

There are therefore exactly two training corpora per condition, ``none`` and ``full``.
Treating swap and back-translation as separate conditions, as the v1 sweep did, measured a
distinction nobody intends to ship: back-translation was always run on top of swap, so
"swap" and "back_translation" were nested rather than alternative.

Both run *after* :func:`data.splits.split_by_source_sentence`, on the train split alone.
Augmenting before splitting would scatter an example and its augmented twin across the
wall, which is the same defect as H5 wearing a different hat.

**Swap can re-create leakage, and v1 could not see it.** Swapping ``(A, B)`` yields a row
whose source sentence is now ``B``. If ``B`` happens to be the source sentence of a dev or
test row, that group has just been bridged. v1 guarded back-translation with an exact
``(original, simplification)`` fingerprint check, which never looks at the source sentence,
so this class of leak was invisible. :func:`assert_augmentation_safe` checks the group.
"""

from __future__ import annotations

import random
import re
from typing import Callable, Iterable, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets

from data.harmonize import normalise_text, pair_key
from data.schema import POLARITY_CLASSES, SYMMETRIC_POLARITIES
from data.splits import LeakageError, group_key

#: Pairs whose two sides are identical gain nothing from being swapped.
SWAP_SKIPPED_SOURCES: frozenset[str] = frozenset({"identical"})

#: Generated identical and unrelated pairs, as a share of the train rows they are built
#: from. Defaults reproduce the v1 proportion (359 of each against 1355 annotated rows),
#: so the augmented condition stays comparable with the published corpus.
DEFAULT_IDENTICAL_RATIO: float = 0.26
DEFAULT_UNRELATED_RATIO: float = 0.26

#: Two sentences sharing more than this fraction of their content tokens are not
#: "unrelated" and are rejected as a generated negative. The v1 article validated its
#: unrelated pairs with ROUGE and BLEU; a token-overlap floor is the same idea, stated
#: as one number that a test can pin down.
MAX_UNRELATED_TOKEN_OVERLAP: float = 0.20


def _rows(dataset: Dataset) -> list[dict]:
    return [dict(zip(dataset.column_names, values)) for values in zip(*(dataset[c] for c in dataset.column_names))]


def swap(dataset: Dataset, forbidden_groups: Optional[set[str]] = None) -> Dataset:
    """Append the mirror of every non-identical pair.

    The label is carried over unchanged: that is the whole point, the metric is meant to be
    symmetric. Identical pairs are skipped because swapping ``(A, A)`` produces ``(A, A)``.

    **The polarity is not carried over in the same way, because it is not symmetric.** If A
    contradicts B then B contradicts A, so a contradiction survives the swap. Entailment
    does not: "a dog is running" entails "an animal is running" and the reverse is false.
    Neither does neutral, since a pair that is neutral one way round can be an entailment
    the other. So a mirrored row keeps its polarity only when that polarity is
    contradiction, and otherwise says it has none, which is a state the schema carries on
    purpose. Copying the label through unchanged would have taught the polarity head that
    entailment is reversible, on every mirrored row of the corpus.

    Swapping moves the simplification into the source-sentence position, so a mirrored row
    can land in a group that belongs to dev or test. Rows that would do so are dropped
    rather than flagged afterwards; this was measured on CSMD, where the check fires on the
    very first fold.

    Args:
        dataset: A train split, harmonised, with ``label`` filled.
        forbidden_groups: Source sentences owned by another split.

    Returns:
        The input followed by its swapped rows, tagged ``source='swapped'``.
    """
    forbidden = forbidden_groups or set()
    seen = {pair_key(o, s) for o, s in zip(dataset["original"], dataset["simplification"])}
    mirrored: list[dict] = []
    for row in _rows(dataset):
        if row["source"] in SWAP_SKIPPED_SOURCES:
            continue
        if group_key(row["simplification"]) in forbidden:
            continue
        key = pair_key(row["simplification"], row["original"])
        if key in seen:
            continue
        seen.add(key)
        keeps_polarity = row.get("polarity_raw", "") in SYMMETRIC_POLARITIES
        mirrored.append(
            {
                **row,
                "item_id": f"{row['item_id']}#swap",
                "original": row["simplification"],
                "simplification": row["original"],
                "source": "swapped",
                "polarity_raw": row.get("polarity_raw", "") if keeps_polarity else "",
                "polarity_scheme": row.get("polarity_scheme", "none") if keeps_polarity else "none",
                "polarity": row.get("polarity", float("nan")) if keeps_polarity else float("nan"),
            }
        )
    if not mirrored:
        return dataset
    return concatenate_datasets([dataset, Dataset.from_list(mirrored).select_columns(dataset.column_names)])


def back_translate(
    dataset: Dataset,
    translate: Callable[[list[str]], list[str]],
    batch_size: int = 32,
    forbidden_groups: Optional[set[str]] = None,
) -> Dataset:
    """Append paraphrased variants of each pair, one per side.

    For a row ``(A, B, label)`` this adds ``(bt(A), B, label)`` and ``(A, bt(B), label)``.
    The label is unchanged, on the assumption that a round trip through another language
    preserves meaning. That assumption is the weak point of this augmentation and it is why
    it stays an option rather than a default.

    Args:
        dataset: A train split.
        translate: Round-trip paraphraser, taking a batch of sentences and returning one
            paraphrase each. Injected so this module needs neither torch nor a network.
        batch_size: Sentences per call to *translate*.
        forbidden_groups: Source sentences owned by another split.

    Returns:
        The input followed by the paraphrased rows, tagged ``source='back_translated'``.
        Rows whose paraphrase is unchanged, which would duplicate an existing pair, or
        whose paraphrased source sentence lands in a held-out group, are dropped.
    """
    forbidden = forbidden_groups or set()
    rows = _rows(dataset)
    originals = [row["original"] for row in rows]
    simplifications = [row["simplification"] for row in rows]

    def run(texts: list[str]) -> list[str]:
        out: list[str] = []
        for start in range(0, len(texts), batch_size):
            out.extend(translate(texts[start : start + batch_size]))
        return out

    bt_originals = run(originals)
    bt_simplifications = run(simplifications)

    seen = {pair_key(o, s) for o, s in zip(originals, simplifications)}
    added: list[dict] = []
    for index, row in enumerate(rows):
        for suffix, original, simplification in (
            ("#bt_o", bt_originals[index], row["simplification"]),
            ("#bt_s", row["original"], bt_simplifications[index]),
        ):
            if not original.strip() or not simplification.strip():
                continue
            if group_key(original) in forbidden:
                continue
            if normalise_text(original) == normalise_text(row["original"]) and suffix == "#bt_o":
                continue
            if normalise_text(simplification) == normalise_text(row["simplification"]) and suffix == "#bt_s":
                continue
            key = pair_key(original, simplification)
            if key in seen:
                continue
            seen.add(key)
            added.append(
                {
                    **row,
                    "item_id": f"{row['item_id']}{suffix}",
                    "original": original,
                    "simplification": simplification,
                    "source": "back_translated",
                }
            )
    if not added:
        return dataset
    return concatenate_datasets([dataset, Dataset.from_list(added).select_columns(dataset.column_names)])


def _content_tokens(text: str) -> set[str]:
    """Lowercased word tokens, used only to measure overlap between two sentences."""
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def token_overlap(left: str, right: str) -> float:
    """Share of the smaller sentence's content tokens that the larger one also has.

    Containment rather than Jaccard: a short sentence fully contained in a long one is not
    unrelated to it, even though their Jaccard similarity would be low.
    """
    a, b = _content_tokens(left), _content_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def generate_identical(dataset: Dataset, ratio: float = DEFAULT_IDENTICAL_RATIO, seed: int = 42) -> Dataset:
    """Append ``(A, A)`` pairs scored 100, built from the split's own source sentences.

    The identical sanity check fails on every run of the sweep. CSMD carries 359 such
    pairs against 1355 annotated ones; this makes that proportion a parameter instead of
    an accident of how the v1 corpus was assembled.

    Args:
        dataset: A train split, harmonised.
        ratio: How many pairs to add, as a share of the input rows.
        seed: Sampling seed.

    Returns:
        The input followed by the generated pairs, tagged ``source='identical'``.
    """
    existing = {pair_key(o, s) for o, s in zip(dataset["original"], dataset["simplification"])}
    candidates = sorted({normalise_text(o) for o in dataset["original"]})
    rng = random.Random(seed)
    rng.shuffle(candidates)

    rows = _rows(dataset)
    template = rows[0]
    added: list[dict] = []
    for index, sentence in enumerate(candidates):
        if len(added) >= int(round(len(dataset) * ratio)):
            break
        key = pair_key(sentence, sentence)
        if key in existing:
            continue
        existing.add(key)
        added.append(
            {
                **template,
                "item_id": f"generated:identical:{index}",
                "original": sentence,
                "simplification": sentence,
                "label": 100.0,
                "label_raw": float("nan"),
                "n_annotators": 0,
                "label_std": float("nan"),
                "source": "identical",
                "system": "generated",
                # Derived, not inherited. ``template`` is simply the first row of the
                # dataset, so copying its polarity would stamp one arbitrary row's class
                # onto every generated pair: if that row happened to be a contradiction,
                # every identical pair would be labelled as one. A sentence trivially
                # entails itself, and that is knowable without an annotator.
                "polarity_raw": "entailment",
                "polarity_scheme": "nli3",
                "polarity": float(POLARITY_CLASSES["entailment"]),
            }
        )
    if not added:
        return dataset
    return concatenate_datasets([dataset, Dataset.from_list(added).select_columns(dataset.column_names)])


def generate_unrelated(
    dataset: Dataset,
    ratio: float = DEFAULT_UNRELATED_RATIO,
    seed: int = 42,
    max_overlap: float = MAX_UNRELATED_TOKEN_OVERLAP,
) -> Dataset:
    """Append ``(A, B)`` pairs scored 0, for sentences that share almost no content.

    ``B`` is drawn from a different source-sentence group and rejected when it shares more
    than *max_overlap* of its content tokens with ``A``. Pairing at random without that
    filter produces some pairs that do share meaning, which would teach the model that a
    genuine paraphrase scores zero.

    Args:
        dataset: A train split, harmonised.
        ratio: How many pairs to add, as a share of the input rows.
        seed: Sampling seed.
        max_overlap: Content-token containment above which a pair is not unrelated.

    Returns:
        The input followed by the generated pairs, tagged ``source='unrelated'``.
    """
    existing = {pair_key(o, s) for o, s in zip(dataset["original"], dataset["simplification"])}
    sentences = sorted({normalise_text(o) for o in dataset["original"]})
    if len(sentences) < 2:
        return dataset

    rng = random.Random(seed)
    rows = _rows(dataset)
    template = rows[0]
    target = int(round(len(dataset) * ratio))
    added: list[dict] = []
    attempts = 0
    # Bounded: a corpus whose sentences all overlap should stop, not spin.
    while len(added) < target and attempts < target * 40:
        attempts += 1
        left, right = rng.sample(sentences, 2)
        key = pair_key(left, right)
        if key in existing or token_overlap(left, right) > max_overlap:
            continue
        existing.add(key)
        added.append(
            {
                **template,
                "item_id": f"generated:unrelated:{len(added)}",
                "original": left,
                "simplification": right,
                "label": 0.0,
                "label_raw": float("nan"),
                "n_annotators": 0,
                "label_std": float("nan"),
                "source": "unrelated",
                "system": "generated",
                # Derived, not inherited, for the same reason as the identical pairs above.
                # Two sentences drawn at random and kept only when their token overlap is
                # below the threshold are neutral: unrelated is not the same as opposed,
                # which is the distinction the whole signed scale rests on.
                "polarity_raw": "neutral",
                "polarity_scheme": "nli3",
                "polarity": float(POLARITY_CLASSES["neutral"]),
            }
        )
    if not added:
        return dataset
    return concatenate_datasets([dataset, Dataset.from_list(added).select_columns(dataset.column_names)])


def assert_augmentation_safe(
    train: Dataset,
    others: Iterable[Dataset],
    baseline: Optional[Dataset] = None,
) -> None:
    """Fail if augmentation *added* a held-out source sentence to the train split.

    Swapping turns the simplification into the source sentence, so a swapped row can land
    in a group that belongs to dev or test even though the row it came from did not. This
    is the check v1 lacked.

    What it deliberately does not do is demand that the split be leak-free to begin with.
    Condition ``a`` of the v2 experiment reproduces v1's row-level split precisely because
    that split leaks; measuring the leak is the point. Comparing against *baseline* keeps
    the check aimed at augmentation, which is the only thing this module controls.

    Args:
        train: The augmented train split.
        others: Every split that must stay disjoint from it.
        baseline: The train split before augmentation. When given, only groups absent from
            it count as a violation.

    Raises:
        LeakageError: Naming the source sentences augmentation brought in.
    """
    held_out: set[str] = set()
    for split in others:
        held_out |= {group_key(o) for o in split["original"]}

    already = {group_key(o) for o in baseline["original"]} if baseline is not None else set()
    offending = sorted(({group_key(o) for o in train["original"]} & held_out) - already)
    if offending:
        raise LeakageError(
            f"augmentation added {len(offending)} held-out source sentence(s) to train, "
            f"e.g. {offending[:2]}. Swapping makes the simplification the source sentence, "
            f"so a swapped row can cross into a dev or test group."
        )


#: The only two training conditions. Everything ships together or nothing does.
MODES: tuple[str, ...] = ("none", "full")


def augment_splits(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    splits: DatasetDict,
    mode: str,
    translate: Optional[Callable[[list[str]], list[str]]] = None,
    batch_size: int = 32,
    identical_ratio: float = DEFAULT_IDENTICAL_RATIO,
    unrelated_ratio: float = DEFAULT_UNRELATED_RATIO,
    seed: int = 42,
) -> tuple[DatasetDict, dict[str, int]]:
    """Augment the train split of *splits*, leaving every other split untouched.

    Order matters. Generation runs first, so the identical and unrelated ratios are taken
    against the real train size rather than against an already inflated one. Swap then
    back-translation follow, as v1 did.

    Back-translating an identical pair yields ``(bt(A), A, 100)``, a paraphrase scored 100.
    v1 did this too. It is a defensible signal and also the one most likely to teach the
    model that lexical divergence is free, which is worth remembering when reading the
    ``none`` versus ``full`` comparison.

    Args:
        splits: Output of :func:`data.splits.split_by_source_sentence`.
        mode: ``none`` or ``full``.
        translate: Required for ``full``.
        batch_size: Sentences per translation call.
        identical_ratio: Generated identical pairs, as a share of the train rows.
        unrelated_ratio: Generated unrelated pairs, as a share of the train rows.
        seed: Sampling seed for generation.

    Returns:
        The augmented splits, and the train row count after each stage, so the three
        contributions stay separable when reading the results.

    Raises:
        ValueError: On an unknown *mode*, or when ``full`` is asked for without a
            translator.
        LeakageError: If augmentation bridged a held-out group.
    """
    if mode not in MODES:
        raise ValueError(f"unknown augmentation mode '{mode}'; expected one of {MODES}")
    if mode == "full" and translate is None:
        raise ValueError("mode 'full' needs a translate callable for back-translation")

    train = splits["train"]
    others = [splits[name] for name in splits if name != "train"]
    forbidden = {group_key(o) for split in others for o in split["original"]}
    counts = {"before": len(train), "after_generation": len(train), "after_swap": len(train), "after": len(train)}

    if mode == "full":
        train = generate_identical(train, ratio=identical_ratio, seed=seed)
        train = generate_unrelated(train, ratio=unrelated_ratio, seed=seed)
        counts["after_generation"] = len(train)
        train = swap(train, forbidden_groups=forbidden)
        counts["after_swap"] = len(train)
        train = back_translate(train, translate, batch_size=batch_size, forbidden_groups=forbidden)
        counts["after"] = len(train)

    augmented = DatasetDict({**{name: splits[name] for name in splits}, "train": train})
    assert_augmentation_safe(train, others, baseline=splits["train"])
    return augmented, counts
