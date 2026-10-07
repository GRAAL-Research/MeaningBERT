"""The negation-lookup baseline: flip the sign when one sentence alone carries a negation.

A reviewer's first question about the signed scale is whether a word list would do as
well: on SICK most contradictions carry an explicit negation. This writes, for the SICK
half of a split, whether exactly one of the two sentences contains a negation word, in
the order the composition dumps its pairs, so the figures generator can score the
lookup beside the polarity heads. Run::

    PYTHONPATH=src python src/diagnostics/negation_lookup.py --out results/v3/negation-lookup-sick-test.json
"""

from __future__ import annotations

import json
import re
from typing import Final

import click

#: Closed list of English negation markers.
NEGATION: Final[re.Pattern] = re.compile(
    r"\b(no|not|nobody|none|nothing|never|nowhere|neither|nor|without)\b|n't", re.IGNORECASE
)


def flags(left: list[str], right: list[str]) -> list[bool]:
    """True when exactly one sentence of the pair carries a negation marker."""
    return [bool(NEGATION.search(a)) != bool(NEGATION.search(b)) for a, b in zip(left, right)]


@click.command()
@click.option("--corpus", default="datastore/polarity/corpus", show_default=True)
@click.option("--split", default="test", show_default=True)
@click.option("--out", required=True)
def main(corpus: str, split: str, out: str) -> None:
    """Write the lookup flags for the SICK half of one split."""
    from datasets import load_from_disk

    rows = load_from_disk(corpus)[split].filter(lambda row: row["corpus"] == "sick")
    got = flags(list(rows["original"]), list(rows["simplification"]))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"split": split, "flags": got}, handle)
    click.echo(f"{sum(got)} / {len(got)} paires signalees : {out}")


if __name__ == "__main__":
    main()
