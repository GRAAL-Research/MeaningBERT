"""v4 magnitude benchmark, beyond French: native human judgements as proxies.

No public corpus rates meaning preservation in many languages (search of 2026-10-21), so
the multilingual reading of V4-H3 rests on two open proxies (decision of 2026-10-21, no
access request, no annotation):

* STS22 (Chen et al., 2022), ten monolingual news sets, rated 1 to 4 by humans on overall
  similarity (MTEB orientation: higher is more similar);
* the TextDetox pairwise content judgements (RUSSE 2022 and CLEF 2024), a binary "same
  meaning" call on a toxic sentence and its detoxified rewrite, in nine languages, grouped
  here by script since the rows carry no language tag.

Similarity and a yes/no call are not graded preservation ratings: the report names them
proxies. Every candidate reads the native text, no translation; none was trained on these
sets. A binary gold makes Pearson the point-biserial correlation.

Run::

    PYTHONPATH=src python src/diagnostics/magnitude_proxies.py --out results/v4/magnitude-proxies.json
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Final

import click
import numpy as np

try:
    from diagnostics.magnitude_fr import bar_name, candidates, compare, correlation, markdown
except ImportError:  # pragma: no cover - run from inside ``src/diagnostics``
    from magnitude_fr import bar_name, candidates, compare, correlation, markdown  # type: ignore

STS22: Final[str] = "mteb/sts22-crosslingual-sts"
STS22_LANGS: Final[tuple[str, ...]] = ("ar", "de", "en", "es", "fr", "it", "pl", "ru", "tr", "zh")
TEXTDETOX: Final[str] = "textdetox/detoxification_pairwise_content_evaluation"
NATIVE: Final[str] = "natif"

#: Unicode ranges telling the TextDetox languages apart; Latin covers en, de and es.
SCRIPTS: Final[list[tuple[str, str]]] = [
    ("cyrillique (ru, uk)", r"[Ѐ-ӿ]"),
    ("arabe", r"[؀-ۿ]"),
    ("devanagari (hi)", r"[ऀ-ॿ]"),
    ("éthiopien (am)", r"[ሀ-፿]"),
    ("chinois", r"[一-鿿]"),
]
LATIN: Final[str] = "latin (en, de, es)"

_PAIR = re.compile(r"text1: (?P<left>.*)\ntext2: (?P<right>.*)\nAnswer:\s*$", re.DOTALL)


def parse_textdetox(prompt: str) -> tuple[str, str]:
    """The evaluated pair of one TextDetox prompt, past its few-shot examples."""
    tail = prompt[prompt.rfind("text1: ") :]
    match = _PAIR.match(tail)
    if match is None:
        raise ValueError(f"prompt TextDetox illisible : {tail[:80]!r}")
    return match["left"].strip(), match["right"].strip()


def script(text: str) -> str:
    """The script group of *text*: the first non-Latin range it uses, else Latin."""
    for name, pattern in SCRIPTS:
        if re.search(pattern, text):
            return name
    return LATIN


def load_sts22(lang: str) -> tuple[list[str], list[str], np.ndarray]:
    """Every split of one STS22 language, as left texts, right texts and gold scores."""
    from datasets import load_dataset

    left: list[str] = []
    right: list[str] = []
    gold: list[float] = []
    for split in load_dataset(STS22, lang).values():
        left += split["sentence1"]
        right += split["sentence2"]
        gold += split["score"]
    return left, right, np.array(gold, dtype=float)


def load_textdetox() -> list[dict[str, Any]]:
    """TextDetox content judgements as ``left``, ``right``, ``label`` (1 = same meaning), ``script``."""
    from datasets import load_dataset

    rows = []
    for split in load_dataset(TEXTDETOX).values():
        for messages in split["messages"]:
            user = next(m["content"] for m in messages if m["role"] == "user")
            answer = next(m["content"] for m in messages if m["role"] == "assistant").strip().lower()
            if answer not in ("yes", "no"):
                raise ValueError(f"reponse TextDetox inattendue : {answer!r}")
            left, right = parse_textdetox(user)
            rows.append({"left": left, "right": right, "label": float(answer == "yes"), "script": script(left)})
    return rows


def block(left: list[str], right: list[str], gold: np.ndarray, trained_root: str) -> dict[str, Any]:
    """Every candidate on one set of native pairs, against the BERTScore bar."""
    scores = candidates(left, right, None, None, "", trained_root, native=NATIVE)
    return {
        "n": int(len(gold)),
        "candidates": {name: {"all": correlation(values, gold)} for name, values in scores.items()},
        "versus_bar": compare(scores, gold, reference=bar_name(NATIVE)),
    }


@click.command()
@click.option("--out", default="results/v4/magnitude-proxies.json", show_default=True)
@click.option("--trained-root", default="results/magnitude-v4/models", show_default=True)
@click.option("--langs", default=",".join(STS22_LANGS), show_default=True, help="Langues STS22.")
@click.option("--textdetox/--no-textdetox", default=True, show_default=True)
def main(out: str, trained_root: str, langs: str, textdetox: bool) -> None:
    """Correlation de chaque magnitude candidate avec les jugements humains de STS22 et TextDetox."""
    result: dict[str, Any] = {}
    for lang in (lang.strip() for lang in langs.split(",") if lang.strip()):
        result[f"STS22 {lang}"] = block(*load_sts22(lang), trained_root)
        click.echo(f"STS22 {lang} : {result[f'STS22 {lang}']['n']} paires")
    if textdetox:
        rows = load_textdetox()
        groups = {"toutes": rows}
        for row in rows:
            groups.setdefault(row["script"], []).append(row)
        for name, group in groups.items():
            gold = np.array([r["label"] for r in group])
            result[f"TextDetox {name}"] = block(
                [r["left"] for r in group], [r["right"] for r in group], gold, trained_root
            )
            click.echo(f"TextDetox {name} : {len(group)} paires")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    with open(os.path.splitext(out)[0] + ".md", "w", encoding="utf-8") as handle:
        for name, section in result.items():
            handle.write(markdown(section, f"## Magnitude sur {name} (proxy, Pearson avec le jugement humain)") + "\n")


if __name__ == "__main__":
    main()
