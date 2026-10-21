"""v4 beyond French: the polarity heads on XNLI in other languages, without new annotation.

XNLI's test split holds the same 5,010 items in fifteen languages, translated by people from
the English ones (Conneau et al., 2018). Each item is therefore paired across languages, and
the same head can be compared with itself in English on exactly the same items: the parity
question of criterion 2, asked in every language instead of French alone.

A multilingual head reads each language directly (zero-shot from its training language). An
English head reads the machine translation to English (translate-test, ``mt-<lang>``) and the
English original (``orig``). The bars mirror criterion 1, per language: the best
off-the-shelf multilingual head reading the language (1a) and the best translate-test (1b).

Caveat the report repeats: the off-the-shelf multilingual heads were trained on machine
translations of MultiNLI, XNLI's source; XNLI test is unseen but in their distribution.

Run, on a GPU host::

    PYTHONPATH=src python src/diagnostics/xnli_multi.py score --heads offshelf-multi \
        --out results/v4/xnli/scores-offshelf-multi.json
    PYTHONPATH=src python src/diagnostics/xnli_multi.py report \
        --scores results/v4/xnli --out results/v4/xnli/report.json
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Final

import click
import numpy as np

try:
    from diagnostics.cross_task import auc
    from diagnostics.discovery_fr import HEAD_LABELS, HEADS, PARITY_MARGIN, p_contradiction
    from diagnostics.significance import delong, holm
except ImportError:  # pragma: no cover - script run from inside ``src/diagnostics``.
    from cross_task import auc  # type: ignore
    from discovery_fr import HEAD_LABELS, HEADS, PARITY_MARGIN, p_contradiction  # type: ignore
    from significance import delong, holm  # type: ignore

#: First wave (decision of 2026-10-21): the two languages of the framing, German and
#: Arabic, plus Spanish and Chinese for four families and three scripts.
LANGS: Final[tuple[str, ...]] = ("de", "ar", "es", "zh")
#: Every XNLI language but English, for the extension.
ALL_LANGS: Final[tuple[str, ...]] = (
    "ar",
    "bg",
    "de",
    "el",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
)
TRANSLATOR: Final[str] = "Helsinki-NLP/opus-mt-{lang}-en"
#: Where Opus-MT has no direct model to English, the multilingual-to-English one stands in.
TRANSLATOR_FALLBACK: Final[str] = "Helsinki-NLP/opus-mt-mul-en"
LANG_LABELS: Final[dict[str, str]] = {
    "en": "EN",
    "ar": "AR",
    "bg": "BG",
    "de": "DE",
    "el": "EL",
    "es": "ES",
    "fr": "FR",
    "hi": "HI",
    "ru": "RU",
    "sw": "SW",
    "th": "TH",
    "tr": "TR",
    "ur": "UR",
    "vi": "VI",
    "zh": "ZH",
}

#: Families read natively; the others (English heads) read the translation.
NATIVE_PREFIXES: Final[tuple[str, ...]] = ("offshelf-mdeberta", "offshelf-xlm", "v4-")

Item = dict[str, Any]


#: Share of items whose label may disagree with English before the pairing is refused.
#: XNLI test in Russian swaps two labels (items 2805 and 2806); a larger share would mean the
#: order itself is lost, not two items.
MAX_MISALIGNED: Final[float] = 0.01


def misaligned(english: list[dict[str, Any]], other: list[dict[str, Any]], lang: str) -> set[int]:
    """Items of *other* whose label disagrees with the English item at the same position.

    Raises:
        ValueError: If the two splits differ in size or disagree on more than
            :data:`MAX_MISALIGNED` of the items.
    """
    if len(english) != len(other):
        raise ValueError(f"XNLI {lang} : {len(other)} items contre {len(english)} en anglais")
    bad = {i for i, (a, b) in enumerate(zip(english, other)) if a["label"] != b["label"]}
    if len(bad) > MAX_MISALIGNED * len(english):
        raise ValueError(f"XNLI {lang} et en ne sont pas alignes item par item ({len(bad)} desaccords)")
    return bad


def load_xnli(langs: tuple[str, ...]) -> tuple[list[Item], dict[str, list[Item]]]:
    """XNLI test in English and in each of *langs*, item by item.

    An item whose label disagrees with English in any of *langs* is dropped from every
    language, so all of them keep exactly the same items.
    """
    from datasets import load_dataset

    english = [dict(row) for row in load_dataset("facebook/xnli", "en", split="test")]
    raw = {lang: [dict(row) for row in load_dataset("facebook/xnli", lang, split="test")] for lang in langs}
    dropped: set[int] = set()
    for lang, rows in raw.items():
        dropped |= misaligned(english, rows, lang)
    keep = [i for i in range(len(english)) if i not in dropped]
    return [english[i] for i in keep], {lang: [rows[i] for i in keep] for lang, rows in raw.items()}


def load_translator(lang: str):
    """The direct Opus-MT model to English, or the multilingual one when none exists."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    for name in (TRANSLATOR.format(lang=lang), TRANSLATOR_FALLBACK):
        try:
            return name, AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name)
        except OSError:
            click.echo(f"{name} introuvable")
    raise click.ClickException(f"aucun traducteur vers l'anglais pour {lang}")


def translate_to_english(sentences: list[str], lang: str, cache_path: str, batch_size: int = 32) -> dict[str, str]:
    """*lang* to English with Opus-MT, cached so every English head reads the same text."""
    cache: dict[str, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            cache = json.load(handle)
    todo = sorted({s for s in sentences if s not in cache}, key=len, reverse=True)
    if todo:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        name, tokenizer, model = load_translator(lang)
        click.echo(f"{lang} : {len(todo)} phrases avec {name}")
        model = model.to(device).eval()
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                generated = model.generate(**encoded, num_beams=4, max_new_tokens=256)
            cache.update(zip(batch, tokenizer.batch_decode(generated, skip_special_tokens=True)))
        del model
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False)
    return cache


def is_native(name: str) -> bool:
    return name.startswith(NATIVE_PREFIXES)


# --------------------------------------------------------------------------- report

LABELS: Final[dict[int, str]] = {0: "entailment", 1: "neutral", 2: "contradiction"}


def _agreement(p: np.ndarray) -> np.ndarray:
    return 1.0 - p


def columns_of(scores: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """One column per head (seeds averaged into ``<family>-ensemble``) and input."""
    families: dict[str, list[str]] = {}
    singles: list[str] = []
    for name in scores:
        match = re.fullmatch(r"(.+)-seed\d+", name)
        if match:
            families.setdefault(match.group(1), []).append(name)
        else:
            singles.append(name)
    out: dict[str, np.ndarray] = {}
    for family, names in families.items():
        for key in sorted({k for n in names for k in scores[n]}):
            stacked = [np.asarray(scores[n][key]) for n in names if key in scores[n]]
            out[f"{family}-ensemble/{key}"] = np.mean(stacked, axis=0)
    for name in singles:
        for key, values in scores[name].items():
            out[f"{name}/{key}"] = np.asarray(values)
    return out


def report(labels: np.ndarray, scores: dict[str, dict[str, Any]], langs: tuple[str, ...]) -> dict[str, Any]:
    """Per language: AUC, parity with English on the same items, tests against the bars."""
    columns = columns_of(scores)
    pos, neg = labels == "entailment", labels == "contradiction"

    def split(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _agreement(values[pos]), _agreement(values[neg])

    aucs = {column: auc(*split(values)) for column, values in columns.items()}
    out: dict[str, Any] = {"n": int(len(labels)), "langs": list(langs), "auc": aucs, "parity": {}, "bars": {}}

    for column, values in columns.items():
        head, key = column.rsplit("/", 1)
        lang = key.removeprefix("mt-")
        if lang not in langs or f"{head}/orig" not in columns:
            continue
        test = delong(*split(columns[f"{head}/orig"]), *split(values))
        se = test["se"]
        out["parity"][column] = {
            "gap": test["diff"],
            "low": test["diff"] - 1.96 * se,
            "high": test["diff"] + 1.96 * se,
            "p_gap_below_margin": (
                float(_normal_cdf((test["diff"] - PARITY_MARGIN) / se))
                if se > 0
                else float(test["diff"] < PARITY_MARGIN)
            ),
        }

    raw: dict[str, float] = {}
    tests: dict[str, dict[str, Any]] = {}
    for lang in langs:
        native = [c for c in columns if c.endswith(f"/{lang}") and c.startswith("offshelf-")]
        translated = [c for c in columns if c.endswith(f"/mt-{lang}") and not c.startswith("v4-")]
        bars = {key: max(pool, key=aucs.get) for key, pool in (("1a", native), ("1b", translated)) if pool}
        out["bars"][lang] = bars
        for column, values in columns.items():
            if not (column.endswith(f"/{lang}") or column.endswith(f"/mt-{lang}")):
                continue
            for key, reference in bars.items():
                if column == reference:
                    continue
                result = delong(*split(values), *split(columns[reference]))
                tests.setdefault(column, {})[key] = result
                raw[f"{column}|{key}"] = result["p"]
    for joined, adjusted in holm(raw).items():
        column, key = joined.split("|")
        tests[column][key]["p_holm"] = adjusted
    out["tests"] = tests
    return out


def _normal_cdf(z: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(z))


def _pct(value: float) -> str:
    return "-" if value is None or np.isnan(value) else f"{100 * value:.2f}"


def _p(value: float) -> str:
    return "-" if value is None or np.isnan(value) else ("< 0.01" if value < 0.01 else f"{value:.2f}")


def _label(head: str) -> str:
    return f"`{HEAD_LABELS.get(head, head)}`"


def tables(result: dict[str, Any]) -> str:
    """AUC per language, parity gap per language, and the tests against the bars."""
    langs = result["langs"]
    heads = sorted({c.rsplit("/", 1)[0] for c in result["auc"]}, key=lambda h: HEAD_LABELS.get(h, h))

    def cell(head: str, lang: str) -> str:
        for key in (lang, f"mt-{lang}"):
            if f"{head}/{key}" in result["auc"]:
                return _pct(result["auc"][f"{head}/{key}"])
        return "-"

    lines = [
        "## XNLI test, AUC par langue (mêmes 5 010 items)",
        "",
        "| Modèle | Lecture | EN | " + " | ".join(LANG_LABELS[lang] for lang in langs) + " |",
        "|---|---|---|" + "---|" * len(langs),
    ]
    for head in heads:
        reading = "langue" if is_native(head) else "traduit"
        english = _pct(result["auc"].get(f"{head}/orig", float("nan")))
        lines.append(
            f"| {_label(head)} | {reading} | {english} | " + " | ".join(cell(head, lang) for lang in langs) + " |"
        )
    lines += [
        "",
        "## Écart avec l'anglais (EN moins langue, points), p de l'écart < 2",
        "",
        "| Modèle | " + " | ".join(LANG_LABELS[lang] for lang in langs) + " |",
        "|---|" + "---|" * len(langs),
    ]
    for head in heads:
        cells = []
        for lang in langs:
            block = next(
                (result["parity"][f"{head}/{k}"] for k in (lang, f"mt-{lang}") if f"{head}/{k}" in result["parity"]),
                None,
            )
            cells.append(f"{_pct(block['gap'])} ({_p(block['p_gap_below_margin'])})" if block else "-")
        lines.append(f"| {_label(head)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Contre les barres de chaque langue (Δ points, p DeLong ajusté Holm)",
        "",
        "| Modèle | " + " | ".join(f"{LANG_LABELS[lang]} vs 1a | {LANG_LABELS[lang]} vs 1b" for lang in langs) + " |",
        "|---|" + "---|---|" * len(langs),
    ]
    for head in heads:
        cells = []
        for lang in langs:
            column = next((f"{head}/{k}" for k in (lang, f"mt-{lang}") if f"{head}/{k}" in result["auc"]), None)
            for key in ("1a", "1b"):
                reference = result["bars"].get(lang, {}).get(key)
                test = result["tests"].get(column or "", {}).get(key)
                if column and column == reference:
                    cells.append("réf.")
                elif test:
                    cells.append(f"{100 * test['diff']:+.2f} ({_p(test['p_holm'])})")
                else:
                    cells.append("-")
        lines.append(f"| {_label(head)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Lecture : langue = la tête lit la langue directement ; traduit = traduction automatique "
        "vers l'anglais (Opus-MT). Les têtes multilingues prêtes à l'emploi ont été entraînées sur "
        "des traductions de MultiNLI, la source d'XNLI.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


@click.group()
def cli() -> None:
    """Tetes de polarite v4 sur XNLI test, dans d'autres langues que le francais."""


def _langs(value: str) -> tuple[str, ...]:
    langs = ALL_LANGS if value == "all" else tuple(v.strip() for v in value.split(",") if v.strip())
    unknown = sorted(set(langs) - set(ALL_LANGS))
    if unknown:
        raise click.BadParameter(f"langues inconnues : {unknown}")
    return langs


@cli.command()
@click.option("--heads", type=click.Choice(sorted(HEADS)), required=True)
@click.option("--out", required=True)
@click.option("--langs", "langs_text", default=",".join(LANGS), show_default=True, help="Liste ou 'all'.")
@click.option("--only", default="")
@click.option("--translations", default="results/v4/xnli/translations", show_default=True)
def score(heads: str, out: str, langs_text: str, only: str, translations: str) -> None:
    """Score XNLI test in English and in each language with one head family, or one head."""
    langs = _langs(langs_text)
    family = [head for head in HEADS[heads] if not only or head[0] == only]
    if not family:
        raise click.ClickException(f"{only} n'est pas une tete de la famille {heads}")
    english, others = load_xnli(langs)
    result: dict[str, Any] = {"n": len(english), "langs": list(langs)}
    tables_mt: dict[str, dict[str, str]] = {}
    for name, checkpoint, _ in family:
        if checkpoint.startswith("results/") and not os.path.isdir(checkpoint):
            click.echo(f"{name} : poids absents ({checkpoint}), tete ignoree")
            continue
        entry: dict[str, Any] = {
            "orig": p_contradiction(
                checkpoint, [r["premise"] for r in english], [r["hypothesis"] for r in english]
            ).tolist()
        }
        for lang, rows in others.items():
            left, right = [r["premise"] for r in rows], [r["hypothesis"] for r in rows]
            if is_native(name):
                entry[lang] = p_contradiction(checkpoint, left, right).tolist()
                continue
            if lang not in tables_mt:
                tables_mt[lang] = translate_to_english(left + right, lang, f"{translations}-{lang}.json")
            table = tables_mt[lang]
            entry[f"mt-{lang}"] = p_contradiction(
                checkpoint, [table[s] for s in left], [table[s] for s in right]
            ).tolist()
        result[name] = entry
        click.echo(f"{name} : {', '.join(entry)}")
    if len(result) == 2:
        raise click.ClickException(f"aucune tete de la famille {heads} n'a ses poids sur cet hote")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle)


@cli.command(name="report")
@click.option("--scores", "scores_dir", required=True)
@click.option("--out", required=True)
@click.option("--langs", "langs_text", default=",".join(LANGS), show_default=True)
def report_command(scores_dir: str, out: str, langs_text: str) -> None:
    """Merge the XNLI score files and write the multilingual report."""

    langs = _langs(langs_text)
    english, _ = load_xnli(langs)
    labels = np.array([LABELS[int(row["label"])] for row in english])
    merged: dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(scores_dir, "scores-*.json"))):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if data.pop("n") != len(labels):
            raise click.ClickException(f"{path} ne porte pas les items d'XNLI test")
        present = set(data.pop("langs"))
        if not set(langs) <= present:
            click.echo(f"{path} : langues manquantes {sorted(set(langs) - present)}, fichier ignore")
            continue
        merged.update(data)
    result = report(labels, merged, langs)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with open(os.path.splitext(out)[0] + ".md", "w", encoding="utf-8") as handle:
        handle.write(tables(result))
    click.echo(f"rapport : {out}")


if __name__ == "__main__":
    cli()
