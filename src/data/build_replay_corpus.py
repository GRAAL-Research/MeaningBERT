"""AUG+LEX plus general inference rows, the v4 regime against forgetting.

The v4 heads keep the lexical guard rails and lose 8 to 13 points of DACCORD against the
off-the-shelf head they start from (``PRODUIT.md``, journal of 2026-10-21): fine-tuning on
VitaminC and the lexical pairs overwrites the general inference the checkpoint came with.
Replay is the standard answer: keep training on rows of the task the checkpoint learned,
here MultiNLI, next to the new ones. Only the train split grows; dev and test stay the
AUG+LEX ones, so model selection and the in-domain numbers remain comparable with the grid.

MultiNLI labels already follow :data:`data.schema.POLARITY_CLASSES` (0 entailment, 1 neutral,
2 contradiction). The rows are balanced per class and drawn with a fixed seed.

Run::

    PYTHONPATH=src python src/data/build_replay_corpus.py \
        --source datastore/polarity-full-lex --out datastore/polarity-full-lex-replay
"""

from __future__ import annotations

import json
from typing import Any, Final

import click

try:
    from data.schema import POLARITY_CLASSES
    from data.translate_polarity_corpus import write
except ImportError:  # pragma: no cover - script run from inside ``src/data``.
    from schema import POLARITY_CLASSES  # type: ignore
    from translate_polarity_corpus import write  # type: ignore

REPLAY_SOURCE: Final[str] = "nyu-mll/multi_nli"
#: MultiNLI machine-translated into 26 languages (Laurer et al.), one ``<lang>_mnli`` split
#: per language: the multilingual replay, the data the off-the-shelf mDeBERTa was trained on.
MULTILINGUAL_SOURCE: Final[str] = "MoritzLaurer/multilingual-NLI-26lang-2mil7"
CLASS_OF: Final[dict[int, str]] = {index: name for name, index in POLARITY_CLASSES.items()}


def replay_rows(rows: list[dict[str, Any]], per_class: int, seed: int, tag: str = "mnli") -> list[dict[str, Any]]:
    """*per_class* MultiNLI rows of each relation, in the polarity corpus schema.

    *tag* names the corpus (``mnli``, or ``mnli-fr`` for a translated split); a row without a
    ``pairID`` is identified by its position in *rows*.

    Raises:
        ValueError: If a relation has fewer than *per_class* rows, rather than silently
            unbalancing the replay.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for index in sorted(CLASS_OF):
        pool = [(position, row) for position, row in enumerate(rows) if int(row["label"]) == index]
        if len(pool) < per_class:
            raise ValueError(f"{CLASS_OF[index]} : {len(pool)} lignes, {per_class} demandees")
        for position in sorted(rng.choice(len(pool), per_class, replace=False).tolist()):
            row_index, row = pool[position]
            out.append(
                {
                    "item_id": f"{tag}:{row.get('pairID', row_index)}",
                    "original": row["premise"],
                    "simplification": row["hypothesis"],
                    "scale": "none",
                    "corpus": tag,
                    "source": "original",
                    "domain": row.get("genre", ""),
                    "system": "mnli",
                    "split_hint": "train",
                    "license": "OANC",
                    "polarity_raw": CLASS_OF[index],
                    "polarity_scheme": "nli3",
                    "label_raw": float("nan"),
                    "label": float("nan"),
                    "label_std": float("nan"),
                    "n_annotators": 1,
                    "polarity": float(index),
                }
            )
    return out


def with_replay(splits, extra: list[dict[str, Any]]):
    """*splits* with *extra* appended to train only, cast to the corpus features."""
    from datasets import Dataset, DatasetDict, concatenate_datasets

    train = splits["train"]
    added = Dataset.from_list(extra).cast(train.features)
    return DatasetDict({**splits, "train": concatenate_datasets([train, added]).shuffle(seed=42)})


@click.command()
@click.option("--source", default="datastore/polarity-full-lex", show_default=True)
@click.option("--out", default="datastore/polarity-full-lex-replay", show_default=True)
@click.option("--per-class", default=30_000, show_default=True, help="MultiNLI rows per relation.")
@click.option("--seed", default=42, show_default=True)
@click.option(
    "--langs",
    default="",
    help="Vide : MultiNLI anglais. Sinon (ex. fr,de,ar,es,zh) : MultiNLI traduit, --per-class reparti par langue.",
)
def main(source: str, out: str, per_class: int, seed: int, langs: str) -> None:
    """Ajoute des paires MultiNLI equilibrees au train du corpus AUG+LEX."""
    from datasets import load_dataset, load_from_disk

    splits = load_from_disk(f"{source}/corpus")
    languages = [lang.strip() for lang in langs.split(",") if lang.strip()]
    if not languages:
        mnli = [row for row in load_dataset(REPLAY_SOURCE, split="train") if int(row["label"]) in CLASS_OF]
        extra = replay_rows(mnli, per_class, seed)
        origin = REPLAY_SOURCE
    else:
        # Same replay size as the English one, shared evenly between the languages.
        extra = []
        for lang in languages:
            rows = [dict(row) for row in load_dataset(MULTILINGUAL_SOURCE, split=f"{lang}_mnli")]
            extra += replay_rows(rows, per_class // len(languages), seed, tag=f"mnli-{lang}")
        origin = MULTILINGUAL_SOURCE
    write(with_replay(splits, extra), source, out)
    with open(f"{out}/replay.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"source": origin, "langs": languages, "per_class": per_class, "seed": seed, "rows": len(extra)}, handle
        )
    click.echo(f"{len(splits['train'])} + {len(extra)} lignes de train : {out}")


if __name__ == "__main__":
    main()
