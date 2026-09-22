"""Build the evaluation suites, with one control pair per annotated pair.

Why. The corpus carries identical and unrelated pairs inherited from v1 and never
regenerated. As v2 tripled the annotated pairs, the coverage collapsed: in ``d_none`` the
test split holds 1536 annotated pairs against 58 identical and 58 unrelated, four percent.
Four percent is enough to notice a catastrophic failure and not enough to measure anything.

What this builds, from the test split of a variant and from it alone, so no sentence
crosses a split boundary:

    identical   (A, A) -> 100, one per distinct source sentence
    unrelated   (A, C) -> 0, one per annotated pair, C taken from another source group
    symmetry    not a dataset. It is the same pairs scored in the other direction, and
                the evaluator mirrors them on the fly. Generating rows for it would only
                duplicate the test set.

The suites are written once and reused, rather than sampled at evaluation time: the
unrelated pairing is random, and two models compared on two different random draws are not
compared at all.

Run::

    PYTHONPATH=src python src/data/build_eval_suites.py --variant-path data/v2/d_none \\
        --out data/v2/eval-suites/d_none
"""

from __future__ import annotations

import os

import click
from datasets import Dataset, DatasetDict, load_from_disk

try:  # PYTHONPATH=src.
    from data.augment import generate_identical, generate_unrelated
except ImportError:  # pragma: no cover
    from augment import generate_identical, generate_unrelated  # type: ignore


def _generated_only(before: Dataset, after: Dataset, source: str) -> Dataset:
    """Keep what the generator appended, drop what it was given.

    The generators return input + generated because that is what the training pipeline
    needs. Here only the new rows are wanted, and they are identified by their tag rather
    than by a row count, so a generator that ever reorders its output cannot corrupt this.
    """
    generated = after.select(range(len(before), len(after)))
    mismatched = [s for s in generated["source"] if s != source]
    if mismatched:
        raise ValueError(f"expected only {source!r} rows, found {sorted(set(mismatched))}")
    return generated


def _identical_both_sides(annotated: Dataset) -> Dataset:
    """``(A, A)`` for every distinct sentence, source side AND simplified side."""
    template = annotated[0]
    seen: set[str] = set()
    rows: list[dict] = []
    for column, side in (("original", "source"), ("simplification", "simplified")):
        for index, text in enumerate(annotated[column]):
            key = " ".join(text.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    **template,
                    "item_id": f"{annotated['item_id'][index]}#identical-{side}",
                    "original": text,
                    "simplification": text,
                    "label": 100.0,
                    "label_raw": 100.0,
                    "source": "identical",
                }
            )
    return Dataset.from_list(rows).select_columns(annotated.column_names)


@click.command()
@click.option("--variant-path", required=True, help="A corpus variant, e.g. data/v2/d_none.")
@click.option("--out", required=True, help="Where the suites are written.")
@click.option("--split", default="test", show_default=True, help="Which split the suites derive from.")
@click.option("--seed", default=42, show_default=True, help="Seed of the unrelated pairing.")
@click.option("--max-overlap", default=0.2, show_default=True,
              help="Content-token containment above which a pair is not unrelated.")
def main(variant_path: str, out: str, split: str, seed: int, max_overlap: float) -> None:
    """Write full-coverage identical and unrelated suites for *variant_path*."""
    data = load_from_disk(variant_path)
    rows = data[split]
    annotated = rows.filter(lambda r: r["source"] == "original")
    print(f"{variant_path} / {split} : {len(rows)} lignes, dont {len(annotated)} annotees")

    # The shared generator builds (A, A) from source sentences only, so it caps at the number
    # of DISTINCT sources: 418 for 1536 annotated pairs in d_none, because several
    # simplifications share one source. Covering the simplified side as well doubles the
    # suite and, more importantly, tests identity on text of a different nature: a
    # simplification is shorter and plainer than its source, and nothing says a model
    # handles both the same way.
    identical = _identical_both_sides(annotated)
    unrelated = _generated_only(
        annotated, generate_unrelated(annotated, ratio=1.0, seed=seed, max_overlap=max_overlap), "unrelated"
    )

    suites = DatasetDict({"identical": identical, "unrelated": unrelated})
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    suites.save_to_disk(out)

    print(f"  identiques  : {len(identical):5d}  ({100 * len(identical) / len(annotated):.0f} % des annotees)")
    print(f"  orthogonales: {len(unrelated):5d}  ({100 * len(unrelated) / len(annotated):.0f} % des annotees)")
    print(f"  symetrie    : {len(annotated):5d}  (les memes paires dans l'autre sens, calculees a l'evaluation)")
    print(f"ecrit dans {out}")


if __name__ == "__main__":
    main()
