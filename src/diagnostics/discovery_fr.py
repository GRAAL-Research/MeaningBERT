"""v4 discovery: does translating to English already give the polarity in French?

The v4 framing (``docs/produit/cadrage-v4-multilingue.md``) makes translate-test the
baseline to beat, and V4-H1 the first hypothesis to check: if translating a French pair to
English and scoring it with the v3 head works as well as a French head could, the v4 has no
reason to exist. This answers it without training anything.

Six French corpora, all already annotated, none ever trained on:

* **DACCORD**, the only one written in French: compatible against contradiction. It alone
  decides criterion 1.
* **SICK-fr**, **Breaking-NLI-fr**, **FraCaS-fr** and **GQNLI-fr** carry their English
  original pair by pair, so the same items can be scored in both languages. **RTE3-FR**
  does not. They are reported and decide nothing, since back-translating a text that
  comes from English often recovers the original and favours translate-test.

Each polarity head reads one of three inputs: ``fr`` (the French pair), ``mt`` (its machine
translation into English, the translate-test) or ``orig`` (the English original, an upper
bound on translate-test). Only ``1 - p_contra`` is kept, the quantity the signed scale
consumes, and a pair falls below zero exactly when ``p_contra > 0.5``, whatever the
magnitude.

Run, on a GPU host::

    PYTHONPATH=src python src/diagnostics/discovery_fr.py score \
        --heads v3 --out results/v4/discovery/scores-v3.json
    PYTHONPATH=src python src/diagnostics/discovery_fr.py report \
        --scores results/v4/discovery --out results/v4/discovery/report.json
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections.abc import Iterable
from typing import Any, Final, Optional

import click
import numpy as np

try:  # PYTHONPATH=src.
    from diagnostics.cross_task import auc
except ImportError:  # pragma: no cover - script run from inside ``src/diagnostics``.
    from cross_task import auc  # type: ignore

#: Hugging Face repositories of the French corpora (Skandalis et al., 2024).
CORPORA: Final[dict[str, tuple[str, str]]] = {
    "daccord": ("maximoss/daccord-contradictions", "daccord_dataset.tsv"),
    "sick_fr": ("maximoss/sick-fr", "SICK_test_fr.tsv"),
    "breaking_fr": ("maximoss/breaking_nli_fr", "breaking_nli_fr.tsv"),
    "rte3_fr": ("maximoss/rte3-french", "RTE3-FR-test-gold-3class.tsv"),
    "gqnli_fr": ("maximoss/gqnli-fr", "gqnli.test.fr.tsv"),
    "fracas_fr": ("maximoss/fracas", "fracas_in_english_and_french.tsv"),
    # XNLI's test split holds the same 5,010 items in every language, in the same order:
    # the paired set criterion 2 (parity with English) is measured on.
    "xnli_fr": ("facebook/xnli", "test"),
}

#: The corpus that decides criterion 1, and the only one written in French.
DECIDING: Final[str] = "daccord"

TRANSLATOR: Final[str] = "Helsinki-NLP/opus-mt-fr-en"

V3_SEEDS: Final[tuple[int, ...]] = tuple(range(42, 52))
V3_TEMPLATE: Final[str] = "results/polarity/nli-deberta-v3-large-fulllex/seed{seed}/model"

#: Head families, each a list of (name, checkpoint, inputs read).
HEADS: Final[dict[str, list[tuple[str, str, tuple[str, ...]]]]] = {
    "v3": [(f"v3-auglex-seed{seed}", V3_TEMPLATE.format(seed=seed), ("mt", "orig", "fr")) for seed in V3_SEEDS],
    "offshelf-en": [
        ("offshelf-deberta-v3-large", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", ("mt", "orig")),
    ],
    "offshelf-multi": [
        ("offshelf-mdeberta-v3-base", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", ("fr", "orig")),
        ("offshelf-xlm-roberta-large", "joeddav/xlm-roberta-large-xnli", ("fr", "orig")),
    ],
    # The v4 grid (run_v4_queue.sh): AUG+LEX on XLM-R-large, trained in English, in
    # translated French, or on both. Read in French for criterion 1, in English for parity.
    **{
        f"v4-{regime}": [
            (
                f"v4-xlmr-{regime}-seed{seed}",
                f"results/polarity/xlmr-large-nli-v4{regime}/seed{seed}/model",
                ("fr", "orig"),
            )
            for seed in V3_SEEDS
        ]
        for regime in ("en", "fr", "enfr")
    },
}

Pair = dict[str, Any]

#: Breaking-NLI categories grouped by the lexical relation the edit introduces. The gold
#: label each family is scored on: contradiction for the first three, entailment for
#: synonyms, whose share below zero is a false alarm.
FAMILIES: Final[dict[str, tuple[tuple[str, ...], str]]] = {
    "antonymes": (("antonyms", "antonyms_wordnet"), "contradiction"),
    "co-hyponymes": (
        (
            "colors",
            "countries",
            "drinks",
            "instruments",
            "materials",
            "nationalities",
            "planets",
            "rooms",
            "vegetables",
        ),
        "contradiction",
    ),
    "nombres": (("cardinals", "ordinals"), "contradiction"),
    "synonymes": (("synonyms",), "entailment"),
}

#: Readable names of the heads in the v4 tables, and of what each head reads.
HEAD_LABELS: Final[dict[str, str]] = {
    "v3-auglex-ensemble": "v3 AUG+LEX",
    "offshelf-deberta-v3-large": "DeBERTa-v3-large NLI",
    "offshelf-mdeberta-v3-base": "mDeBERTa-v3-base NLI",
    "offshelf-xlm-roberta-large": "XLM-R-large NLI",
    "v4-xlmr-en-ensemble": "XLM-R-large AUG+LEX (EN)",
    "v4-xlmr-fr-ensemble": "XLM-R-large AUG+LEX (FR traduit)",
    "v4-xlmr-enfr-ensemble": "XLM-R-large AUG+LEX (EN+FR)",
}
INPUT_LABELS: Final[dict[str, str]] = {"fr": "français", "mt": "traduit", "orig": "anglais d'origine"}
CORPUS_LABELS: Final[dict[str, str]] = {
    "daccord": "DACCORD",
    "sick_fr": "SICK-fr",
    "breaking_fr": "Breaking-fr",
    "rte3_fr": "RTE3-FR",
    "fracas_fr": "FraCaS-fr",
    "gqnli_fr": "GQNLI-fr",
    "xnli_fr": "XNLI-fr",
}


def _rows(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))


def _nli(label: str) -> Optional[str]:
    """Map the label spellings of the corpora onto the three relations, or drop the row."""
    text = label.strip().lower()
    mapping = {
        "0": "entailment",
        "1": "neutral",
        "2": "contradiction",
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
    }
    return mapping.get(text)


def _fracas_label(row: dict[str, str]) -> Optional[str]:
    answer = row.get("answer", "").strip().lower()
    return {"yes": "entailment", "no": "contradiction", "unknown": "neutral", "don't know": "neutral"}.get(answer)


def normalize(
    corpus: str, rows: Iterable[dict[str, str]], seen: frozenset[tuple[str, str]] = frozenset()
) -> list[Pair]:
    """One shape for every corpus: French pair, English original if any, relation.

    Args:
        corpus: Key of :data:`CORPORA`.
        rows: Raw TSV rows.
        seen: English pairs the v3 head was trained or tuned on, dropped from SICK-fr.

    Returns:
        Pairs with ``fr_a``, ``fr_b``, ``en_a``, ``en_b`` (empty when absent), ``label`` and
        ``category``.
    """
    pairs: list[Pair] = []
    for row in rows:
        en_a = en_b = ""
        category = ""
        if corpus == "daccord":
            fr_a, fr_b = row["premise"], row["hypothesis"]
            label: Optional[str] = "contradiction" if row["label"].strip() == "1" else "compatible"
            category = row.get("genre", "")
        elif corpus == "sick_fr":
            fr_a, fr_b = row["sentence_A"], row["sentence_B"]
            en_a, en_b = row["original_SICK_sentence_A"], row["original_SICK_sentence_B"]
            label = _nli(row["entailment_label"])
            if (en_a.strip(), en_b.strip()) in seen:
                continue
        elif corpus == "breaking_fr":
            fr_a, fr_b = row["sentence1_fr"], row["sentence2_fr"]
            en_a, en_b = row["sentence1_en_orig"], row["sentence2_en_orig"]
            label = _nli(row["gold_label"])
            category = row["category"]
        elif corpus == "rte3_fr":
            fr_a, fr_b = row["premise"], row["hypothesis"]
            label = _nli(row["label_text"])
        elif corpus == "gqnli_fr":
            fr_a, fr_b = row["premise"], row["hypothesis"]
            en_a, en_b = row["premise_original"], row["hypothesis_original"]
            label = _nli(row["label_text"])
        elif corpus == "fracas_fr":
            fr_a, fr_b = row["premises"], row["hypothesis"]
            en_a, en_b = row["premises_original"], row["hypothesis_original"]
            label = _fracas_label(row)
        elif corpus == "xnli_fr":
            fr_a, fr_b = row["premise"], row["hypothesis"]
            en_a, en_b = row["premise_en"], row["hypothesis_en"]
            label = _nli(str(row["label"]))
        else:
            raise ValueError(f"unknown corpus {corpus!r}")
        if label is None or not fr_a.strip() or not fr_b.strip():
            continue
        pairs.append(
            {
                "corpus": corpus,
                "fr_a": fr_a.strip(),
                "fr_b": fr_b.strip(),
                "en_a": en_a.strip(),
                "en_b": en_b.strip(),
                "label": label,
                "category": category,
            }
        )
    return pairs


def xnli_rows() -> list[dict[str, str]]:
    """XNLI test in French, each row carrying its English counterpart.

    Raises:
        ValueError: If the two languages do not hold the same labels in the same order,
            the one property the pairing rests on.
    """
    from datasets import load_dataset

    french = load_dataset("facebook/xnli", "fr", split="test")
    english = load_dataset("facebook/xnli", "en", split="test")
    if french["label"] != english["label"]:
        raise ValueError("XNLI fr et en ne sont pas alignes item par item")
    return [
        {
            "premise": fr["premise"],
            "hypothesis": fr["hypothesis"],
            "premise_en": en["premise"],
            "hypothesis_en": en["hypothesis"],
            "label": str(fr["label"]),
        }
        for fr, en in zip(french, english)
    ]


def load_corpora(seen: frozenset[tuple[str, str]] = frozenset()) -> list[Pair]:
    """Download and normalize the seven corpora."""
    from huggingface_hub import hf_hub_download

    pairs: list[Pair] = []
    for name, (repo, filename) in CORPORA.items():
        if name == "xnli_fr":
            pairs.extend(normalize(name, xnli_rows(), seen))
            continue
        path = hf_hub_download(repo, filename, repo_type="dataset")
        pairs.extend(normalize(name, _rows(path), seen))
    return pairs


def v3_seen_pairs(corpus_path: str) -> frozenset[tuple[str, str]]:
    """English pairs of the v3 train and dev splits, in both orders."""
    from datasets import load_from_disk

    data = load_from_disk(corpus_path)
    seen: set[tuple[str, str]] = set()
    for split in ("train", "dev"):
        for left, right in zip(data[split]["original"], data[split]["simplification"]):
            seen.add((left.strip(), right.strip()))
            seen.add((right.strip(), left.strip()))
    return frozenset(seen)


def translate(sentences: list[str], cache_path: str, batch_size: int = 32) -> dict[str, str]:
    """French to English with :data:`TRANSLATOR`, cached so every head reads the same text."""
    cache: dict[str, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            cache = json.load(handle)
    todo = sorted({s for s in sentences if s not in cache})
    if todo:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR)
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR).to(device).eval()
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                generated = model.generate(**encoded, num_beams=4, max_new_tokens=512)
            for source, target in zip(batch, tokenizer.batch_decode(generated, skip_special_tokens=True)):
                cache[source] = target
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False)
    return cache


def contradiction_index(id2label: dict[Any, str]) -> int:
    """Index of the contradiction logit, read from the config and never assumed.

    The off-the-shelf heads do not share an order: ``joeddav/xlm-roberta-large-xnli`` puts
    contradiction first, the v3 heads and ``MoritzLaurer`` checkpoints last.
    """
    matches = [int(index) for index, name in id2label.items() if str(name).lower().startswith("contra")]
    if len(matches) != 1:
        raise ValueError(f"no single contradiction label in {id2label!r}")
    return matches[0]


def p_contradiction(checkpoint: str, left: list[str], right: list[str], batch_size: int = 16) -> np.ndarray:
    """Probability of contradiction for each pair under *checkpoint*."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
    index = contradiction_index(model.config.id2label)
    out: list[np.ndarray] = []
    for start in range(0, len(left), batch_size):
        encoded = tokenizer(
            left[start : start + batch_size],
            right[start : start + batch_size],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out.append(model(**encoded).logits.float().softmax(-1)[:, index].cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out) if out else np.zeros(0)


def inputs_for(pairs: list[Pair], kind: str, translations: dict[str, str]) -> tuple[list[int], list[str], list[str]]:
    """Indices and sentence lists a head reads for input *kind*."""
    if kind == "fr":
        keep = list(range(len(pairs)))
        return keep, [pairs[i]["fr_a"] for i in keep], [pairs[i]["fr_b"] for i in keep]
    if kind == "mt":
        keep = list(range(len(pairs)))
        return keep, [translations[pairs[i]["fr_a"]] for i in keep], [translations[pairs[i]["fr_b"]] for i in keep]
    if kind == "orig":
        keep = [i for i, pair in enumerate(pairs) if pair["en_a"] and pair["en_b"]]
        return keep, [pairs[i]["en_a"] for i in keep], [pairs[i]["en_b"] for i in keep]
    raise ValueError(f"unknown input {kind!r}")


# --------------------------------------------------------------------------- report


def _below_zero(p: np.ndarray) -> np.ndarray:
    """The signed score is negative exactly when ``1 - 2 p_contra < 0``."""
    return p > 0.5


def bootstrap_auc_gap(
    positive_a: np.ndarray,
    negative_a: np.ndarray,
    positive_b: np.ndarray,
    negative_b: np.ndarray,
    rounds: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap of ``AUC(a) - AUC(b)`` over the same items.

    Both heads score the same pairs, so items are resampled once and the two AUCs are
    recomputed on the same draw; resampling them independently would widen the interval
    with variance the comparison does not have.
    """
    rng = np.random.default_rng(seed)
    gaps = np.empty(rounds)
    n_pos, n_neg = len(positive_a), len(negative_a)
    for draw in range(rounds):
        pos = rng.integers(0, n_pos, n_pos)
        neg = rng.integers(0, n_neg, n_neg)
        gaps[draw] = auc(positive_a[pos], negative_a[neg]) - auc(positive_b[pos], negative_b[neg])
    gap = auc(positive_a, negative_a) - auc(positive_b, negative_b)
    low, high = np.percentile(gaps, [2.5, 97.5])
    return {"gap": float(gap), "low": float(low), "high": float(high)}


def _agreement(p: np.ndarray) -> np.ndarray:
    return 1.0 - p


def slice_report(p: np.ndarray, labels: np.ndarray, positive: str) -> dict[str, Any]:
    """AUC of ``1 - p_contra`` (*positive* above contradiction) and the share below zero."""
    pos, neg = labels == positive, labels == "contradiction"
    out: dict[str, Any] = {
        "n": int(len(labels)),
        "auc": auc(_agreement(p[pos]), _agreement(p[neg])) if pos.any() and neg.any() else float("nan"),
    }
    for name in sorted(set(labels.tolist())):
        keep = labels == name
        out[f"below_zero_{name}"] = float(_below_zero(p[keep]).mean())
    return out


def report(pairs: list[Pair], scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per corpus, per head and input: AUC and shares below zero; then criterion 1."""
    corpora = np.array([pair["corpus"] for pair in pairs])
    labels = np.array([pair["label"] for pair in pairs])
    categories = np.array([pair["category"] for pair in pairs])

    def full(name: str, key: str) -> Optional[np.ndarray]:
        """Scores of *name* on *key*, scattered back to one value per pair (NaN if unread)."""
        if key not in scores.get(name, {}):
            return None
        values = np.full(len(pairs), np.nan)
        values[np.array(scores[name][f"{key}_index"], dtype=int)] = scores[name][key]
        return values

    # Seeds of one trained family average into one column, ``<family>-ensemble``.
    families: dict[str, list[str]] = {}
    for name in scores:
        match = re.fullmatch(r"(.+)-seed\d+", name)
        if match:
            families.setdefault(match.group(1), []).append(name)
    v3_names = families.get("v3-auglex", [])
    columns: dict[str, np.ndarray] = {}
    for family, names in families.items():
        for key in ("mt", "orig", "fr"):
            stacked = [s for s in (full(name, key) for name in names) if s is not None]
            if stacked:
                columns[f"{family}-ensemble/{key}"] = np.mean(stacked, axis=0)
    seeded = {name for names in families.values() for name in names}
    for name in scores:
        if name in seeded:
            continue
        for key in ("mt", "orig", "fr"):
            values = full(name, key)
            if values is not None:
                columns[f"{name}/{key}"] = values
    out_seeds = {family: len(names) for family, names in families.items()}

    out: dict[str, Any] = {
        "n_pairs": {c: int((corpora == c).sum()) for c in CORPORA},
        "seeds": out_seeds,
        "corpora": {},
    }
    for corpus in CORPORA:
        keep = corpora == corpus
        positive = "compatible" if corpus == DECIDING else "entailment"
        block: dict[str, Any] = {}
        for column, values in columns.items():
            usable = keep & ~np.isnan(values)
            if usable.any():
                block[column] = slice_report(values[usable], labels[usable], positive)
                if corpus == "breaking_fr":
                    block[column]["per_family_below_zero"] = {
                        family: float(_below_zero(values[usable & np.isin(categories, cats) & (labels == gold)]).mean())
                        for family, (cats, gold) in FAMILIES.items()
                        if (usable & np.isin(categories, cats) & (labels == gold)).any()
                    }
        out["corpora"][corpus] = block

    # Per-seed spread of the translate-test AUC on the deciding corpus.
    keep = corpora == DECIDING
    seed_aucs = []
    for name in v3_names:
        values = full(name, "mt")
        if values is not None:
            seed_aucs.append(slice_report(values[keep], labels[keep], "compatible")["auc"])
    out["v3_translate_test_daccord_auc_per_seed"] = {
        "mean": float(np.mean(seed_aucs)) if seed_aucs else float("nan"),
        "std": float(np.std(seed_aucs, ddof=1)) if len(seed_aucs) > 1 else float("nan"),
    }

    # Criterion 1, split in two (decision of 2026-10-11). 1a: beat the best off-the-shelf
    # multilingual head reading French, which decides whether a French head is worth
    # building. 1b: beat the best translate-test, which decides what the article may claim.
    # Each bar is the best measured candidate, never one picked after seeing the French head.
    daccord_auc = {
        column: auc(
            _agreement(values[keep & (labels == "compatible")]), _agreement(values[keep & (labels == "contradiction")])
        )
        for column, values in columns.items()
        if not np.isnan(values[keep]).any()
    }
    candidates = {
        "1a": [c for c in daccord_auc if c.endswith("/fr") and c.startswith("offshelf-")],
        "1b": [c for c in daccord_auc if c.endswith("/mt") and not c.startswith("v4-")],
    }
    pos, neg = keep & (labels == "compatible"), keep & (labels == "contradiction")
    criterion: dict[str, Any] = {}
    for key, pool in candidates.items():
        if not pool:
            continue
        best = max(pool, key=lambda c: daccord_auc[c])
        reference = columns[best]
        criterion[key] = {
            "reference": best,
            "auc": daccord_auc[best],
            "gaps": {
                column: bootstrap_auc_gap(
                    _agreement(columns[column][pos]),
                    _agreement(columns[column][neg]),
                    _agreement(reference[pos]),
                    _agreement(reference[neg]),
                )
                for column in daccord_auc
                if column != best
            },
        }
    out["criterion_1"] = criterion
    return out


def readable(column: str) -> tuple[str, str]:
    """Readable (model, input) for a report column such as ``offshelf-xlm-roberta-large/fr``."""
    head, kind = column.rsplit("/", 1)
    return f"`{HEAD_LABELS.get(head, head)}`", INPUT_LABELS.get(kind, kind)


def _pct(value: float) -> str:
    return "-" if value is None or np.isnan(value) else f"{100 * value:.2f}"


def tables(result: dict[str, Any]) -> str:
    """The discovery report as three short Markdown tables."""
    daccord = result["corpora"][DECIDING]
    order = sorted(daccord, key=lambda c: -daccord[c]["auc"])
    criterion = result.get("criterion_1", {})
    bars = {criterion[k]["reference"]: k for k in criterion}

    lines = [
        "## DACCORD, le seul corpus écrit en français",
        "",
        "| Modèle | Entrée | AUC | Contradictions < 0 | Compatibles < 0 | Barre |",
        "|---|---|---|---|---|---|",
    ]
    for column in order:
        model, kind = readable(column)
        m = daccord[column]
        lines.append(
            f"| {model} | {kind} | {_pct(m['auc'])} | {_pct(m.get('below_zero_contradiction'))} "
            f"| {_pct(m.get('below_zero_compatible'))} | {bars.get(column, '')} |"
        )
    lines += [
        "",
        "Barre 1a : meilleure tête multilingue prête à l'emploi qui lit le français. "
        "Barre 1b : meilleur translate-test.",
        "",
        "## AUC par corpus",
        "",
        "| Modèle | Entrée | " + " | ".join(CORPUS_LABELS[c] for c in CORPORA) + " |",
        "|---|---|" + "---|" * len(CORPORA),
    ]
    columns = sorted({c for block in result["corpora"].values() for c in block}, key=readable)
    for column in columns:
        model, kind = readable(column)
        cells = [_pct(result["corpora"][corpus].get(column, {}).get("auc", float("nan"))) for corpus in CORPORA]
        lines.append(f"| {model} | {kind} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Breaking-fr, part sous zéro par famille lexicale",
        "",
        "| Modèle | Entrée | " + " | ".join(FAMILIES) + " |",
        "|---|---|" + "---|" * len(FAMILIES),
    ]
    breaking = result["corpora"]["breaking_fr"]
    for column in sorted(breaking, key=readable):
        model, kind = readable(column)
        family = breaking[column].get("per_family_below_zero", {})
        lines.append(f"| {model} | {kind} | " + " | ".join(_pct(family.get(f, float("nan"))) for f in FAMILIES) + " |")
    lines += [
        "",
        "Synonymes sous zéro = fausses alarmes. Co-hyponymes : Breaking-NLI les étiquette "
        "contradiction, la v3 est conçue pour les garder au-dessus de zéro.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


@click.group()
def cli() -> None:
    """Discovery v4 : translate-test contre tetes multilingues, sur corpus francais."""


@cli.command()
@click.option("--heads", type=click.Choice(sorted(HEADS)), required=True)
@click.option("--out", required=True, help="JSON of p_contra per head and input.")
@click.option("--v3-corpus", default="datastore/polarity-full-lex/corpus", show_default=True)
@click.option("--translations", default="results/v4/discovery/translations.json", show_default=True)
@click.option("--only", default="", help="Score only this head of the family, e.g. v4-xlmr-en-seed42.")
def score(heads: str, out: str, v3_corpus: str, translations: str, only: str) -> None:
    """Score every French pair with one head family, or one head of it."""
    family = [head for head in HEADS[heads] if not only or head[0] == only]
    if not family:
        raise click.ClickException(f"{only} n'est pas une tete de la famille {heads}")
    seen = v3_seen_pairs(v3_corpus) if os.path.exists(v3_corpus) else frozenset()
    if not seen:
        raise click.ClickException(f"corpus v3 introuvable : {v3_corpus} ; sans lui, SICK-fr fuit")
    pairs = load_corpora(seen)
    needs_mt = any("mt" in inputs for _, _, inputs in family)
    sentences = [s for pair in pairs for s in (pair["fr_a"], pair["fr_b"])]
    table = translate(sentences, translations) if needs_mt else {}

    result: dict[str, Any] = {"n_pairs": len(pairs)}
    for name, checkpoint, kinds in family:
        # A local seed not copied to this host is skipped out loud, never silently: the
        # report's per-seed spread says how many seeds it rests on.
        if checkpoint.startswith("results/") and not os.path.isdir(checkpoint):
            click.echo(f"{name} : poids absents ({checkpoint}), tete ignoree")
            continue
        entry: dict[str, Any] = {"checkpoint": checkpoint}
        for kind in kinds:
            index, left, right = inputs_for(pairs, kind, table)
            entry[kind] = [float(v) for v in p_contradiction(checkpoint, left, right)]
            entry[f"{kind}_index"] = index
        result[name] = entry
        click.echo(f"{name} : {', '.join(kinds)}")
    scored = len(result) - 1
    if not scored:
        raise click.ClickException(f"aucune tete de la famille {heads} n'a ses poids sur cet hote")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    click.echo(f"{len(pairs)} paires, {scored} tetes : {out}")


@cli.command(name="report")
@click.option("--scores", "scores_dir", required=True, help="Folder of scores-*.json files.")
@click.option("--out", required=True)
@click.option("--v3-corpus", default="datastore/polarity-full-lex/corpus", show_default=True)
def report_command(scores_dir: str, out: str, v3_corpus: str) -> None:
    """Merge the score files and write the discovery report."""
    pairs = load_corpora(v3_seen_pairs(v3_corpus))
    merged: dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(scores_dir, "scores-*.json"))):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if data.pop("n_pairs") != len(pairs):
            raise click.ClickException(f"{path} ne porte pas les memes paires que le rapport")
        merged.update(data)
    result = report(pairs, merged)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    markdown = os.path.splitext(out)[0] + ".md"
    with open(markdown, "w", encoding="utf-8") as handle:
        handle.write(tables(result))
    click.echo(f"rapport : {out} ; tableaux : {markdown}")


if __name__ == "__main__":
    cli()
