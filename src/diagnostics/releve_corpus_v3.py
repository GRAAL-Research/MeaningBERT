"""Check which polarity corpora are actually loadable, before planning anything on them.

Why this runs first. The v2 campaign was decided by its corpus survey, not by its models:
SALSA turned out to be a dead dataset published only as 50 demonstration pairs, PLABA
needed a new scale added to the contract, and the encoder in the end accounted for +0.067
of Pearson against +0.314 for the data. Planning a v3 on corpora nobody has opened is how
that lesson gets unlearned.

For each candidate this reports what a plan actually needs: does it load, how many rows,
which columns carry the pair and the label, what the label space looks like, and what the
licence permits. A corpus that fails here is not a risk to manage later, it is a corpus to
replace now.

Run::

    PYTHONPATH=src python src/diagnostics/releve_corpus_v3.py --out docs/corpus-v3-releve.md
"""

from __future__ import annotations

import json
from typing import Optional

import click

#: The polarity candidates from ROADMAP.md, in the order the roadmap ranks them.
#:
#: ``config`` is the HuggingFace configuration name when the dataset needs one. ``why``
#: records what each is meant to contribute, so a corpus that fails can be replaced by
#: something that fills the same hole rather than by whatever is next on the list.
CANDIDATES = [
    ("tals/vitaminc", None, "450k contrastive claim-evidence pairs; the largest minimal-pair corpus there is"),
    ("nightingal3/fig-qa", None, "control: figurative pairs, to see whether the harness reads an unrelated schema"),
    ("EdinburghNLP/ACES", None, "36k examples over 68 phenomena: negation, antonyms, numbers, entities, argument order"),
    ("google-research-datasets/paws", "labeled_final", "108k pairs with high lexical overlap that are NOT paraphrases"),
    ("sentence-transformers/stsb", None, "continuous relatedness, the bridge between a magnitude head and a polarity head"),
    ("sick", None, "relatedness AND entailment on the same pairs; the natural bridge between the two heads"),
    ("pietrolesci/nan-nli", None, "sub-clausal negation, targeted and small"),
    ("sagnikrayc/monli", None, "downward monotonicity under negation"),
    ("lasha-nlp/CondaQA", None, "negation scope in reading comprehension"),
]


def probe(name: str, config: Optional[str]) -> dict:
    """Open one dataset and report what a plan would need to know about it.

    Loading only the first split and only its metadata keeps this cheap: the point is to
    learn whether the corpus exists and what shape it has, not to download 450k rows.
    """
    from datasets import get_dataset_config_names, load_dataset_builder

    out: dict = {"name": name, "config": config}
    try:
        if config is None:
            configs = get_dataset_config_names(name)
            out["configs"] = configs[:6]
            config = configs[0] if configs else None
            out["config"] = config
        builder = load_dataset_builder(name, config)
        info = builder.info
        out["ok"] = True
        out["rows"] = {k: v.num_examples for k, v in (info.splits or {}).items()}
        out["columns"] = sorted(info.features.keys()) if info.features else []
        out["licence"] = info.license or "non declaree"
        label = next((f for f in ("label", "gold_label", "labels") if info.features and f in info.features), None)
        out["label_field"] = label
        if label is not None:
            names = getattr(info.features[label], "names", None)
            out["label_space"] = names if names else str(info.features[label])
    except Exception as exc:  # noqa: BLE001 - the failure IS the result here
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return out


@click.command()
@click.option("--out", default=None, help="Markdown report to write, in addition to stdout.")
@click.option("--json-out", default=None, help="Raw findings, for the planning scripts.")
def main(out: Optional[str], json_out: Optional[str]) -> None:
    """Probe every v3 candidate and report what is usable."""
    findings = []
    for name, config, why in CANDIDATES:
        print(f"--- {name}")
        found = probe(name, config)
        found["why"] = why
        findings.append(found)
        if found["ok"]:
            total = sum(found.get("rows", {}).values())
            print(f"    {total:>9,} lignes  |  {found['licence']}  |  etiquette : {found.get('label_space', 'aucune')}")
            print(f"    colonnes : {', '.join(found['columns'][:8])}")
        else:
            print(f"    INDISPONIBLE  {found['error']}")

    usable = [f for f in findings if f["ok"]]
    print(f"\n{len(usable)} corpus sur {len(findings)} sont ouvrables, "
          f"{sum(sum(f.get('rows', {}).values()) for f in usable):,} lignes au total")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(findings, handle, indent=2)
        print(f"brut : {json_out}")

    if out:
        lines = [
            "# Releve de disponibilite, corpus de polarite (v3)",
            "",
            "Genere par `src/diagnostics/releve_corpus_v3.py`. Ne pas editer a la main : un releve",
            "ecrit a la main pourrit en silence, ce qui est la pire facon d'avoir tort.",
            "",
            "| corpus | etat | lignes | etiquette | licence | role |",
            "|---|---|---|---|---|---|",
        ]
        for f in findings:
            if f["ok"]:
                total = f"{sum(f.get('rows', {}).values()):,}"
                lines.append(f"| `{f['name']}` | ouvrable | {total} | {f.get('label_space', '--')} "
                             f"| {f['licence']} | {f['why']} |")
            else:
                lines.append(f"| `{f['name']}` | **indisponible** | -- | -- | -- | {f['why']} |")
        lines += ["", "## Echecs", ""]
        for f in findings:
            if not f["ok"]:
                lines.append(f"- `{f['name']}` : {f['error']}")
        with open(out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        print(f"rapport : {out}")


if __name__ == "__main__":
    main()
