"""Minimal-pair table: share of pairs that land on the right side of zero, per kind of edit.

Each row is a scale (the negation lookup, the off-the-shelf head, the fine-tuned heads),
each column a kind of one-word edit from ``src/diagnostics/minimal_pairs.py``. A
contradiction (negation, antonym) is right below zero; a preserved meaning (synonym,
hypernym, co-hyponym) is right above it. Antonyms are split by whether their word pair
belongs to the half of the lexicon the lexical augmentation trained on, so a gain on the
unseen half cannot come from having seen the pair. Run::

    PYTHONPATH=src python src/figures_generator/minimal_pairs_table.py --tex-out paper/v3
"""

from __future__ import annotations

import glob
import json
import os
from typing import Final, Optional

import click
import numpy as np

try:
    from data.build_lexical_augmentation import test_words, train_tables
    from diagnostics.negation_lookup import NEGATION
except ImportError:  # pragma: no cover - run from the repository root
    from src.data.build_lexical_augmentation import test_words, train_tables  # type: ignore
    from src.diagnostics.negation_lookup import NEGATION  # type: ignore

#: Column: (kind, which half of the lexicon, the side of zero that is right).
COLUMNS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("Negation", "negation", "all", "below"),
    ("Antonym, seen", "antonym", "seen", "below"),
    ("Antonym, unseen", "antonym", "unseen", "below"),
    ("Synonym", "synonym", "all", "above"),
    ("Hypernym", "hypernym", "all", "above"),
    ("Co-hyponym", "cohyponym", "all", "above"),
)

HEAD_WORDS: Final[dict[int, str]] = {10: "ten"}


def lexicon_half(pair: dict) -> str:
    """``seen`` when the training half of the lexical augmentation holds the edited word."""
    if pair["kind"] not in ("antonym", "synonym", "hypernym"):
        return "all"
    antonyms, entailing = train_tables()
    old = pair["old"].lower()
    if old in antonyms or old in entailing:
        return "seen"
    return "unseen" if old in test_words() else "seen"


def masks(pairs: list[dict]) -> dict[str, np.ndarray]:
    """Boolean mask per column."""
    kind = np.array([pair["kind"] for pair in pairs])
    half = np.array([lexicon_half(pair) for pair in pairs])
    out = {}
    for name, edit, which, _side in COLUMNS:
        mask = kind == edit
        if which != "all":
            mask &= half == which
        out[name] = mask
    return out


def right_side(signed: np.ndarray, mask: np.ndarray, side: str) -> float:
    """Percent of the masked pairs on the expected side of zero."""
    values = signed[mask]
    if values.size == 0:
        return float("nan")
    return float(100 * ((values < 0) if side == "below" else (values > 0)).mean())


def cell(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{np.mean(values):.2f}$_{{\\pm {np.std(values, ddof=1):.2f}}}$"


def minimal_table(pairs: list[dict], scales: dict[str, list[np.ndarray]], path: str) -> None:
    """One row per scale; several arrays in a row are heads, reported as mean and spread."""
    got = masks(pairs)
    counts = " & ".join(str(int(got[name].sum())) for name, *_ in COLUMNS)
    lines = [
        r"% Genere par src/figures_generator/minimal_pairs_table.py. Ne pas editer a la main.",
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\begin{tabular}{l " + "c" * len(COLUMNS) + "}",
        r"\toprule",
        "Edit & " + " & ".join(name for name, *_ in COLUMNS) + r" \\",
        r"Expected sign & $<0$ & $<0$ & $<0$ & $>0$ & $>0$ & $>0$ \\",
        r"Pairs & " + counts + r" \\",
        r"\midrule",
    ]
    for label, heads in scales.items():
        cells = [cell([right_side(h, got[name], side) for h in heads]) for name, _e, _w, side in COLUMNS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{One-word edits of SICK test sentences: percent of pairs on the expected side of zero. "
        r"Ours: mean with standard deviation as subscript over ten heads. Unseen antonyms are held out of "
        r"\textsc{lex}.}",
        r"\label{tab:minimal}",
        r"\end{table*}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def lookup_signed(pairs: list[dict], magnitude: np.ndarray) -> np.ndarray:
    flags = np.array([bool(NEGATION.search(p["source"])) != bool(NEGATION.search(p["edited"])) for p in pairs])
    return np.where(flags, -magnitude, magnitude)


def _signed(paths: list[str]) -> list[np.ndarray]:
    out = []
    for path in sorted(paths):
        with open(path, encoding="utf-8") as handle:
            out.append(np.array(json.load(handle)["signed"], dtype=float))
    return out


@click.command()
@click.option("--pairs", "pairs_path", default="results/v3/minimal-pairs.json", show_default=True)
@click.option("--scores", "scores_dir", default="results/v3/minimal", show_default=True)
@click.option("--tex-out", default="paper/v3", show_default=True)
def main(pairs_path: str, scores_dir: str, tex_out: Optional[str]) -> None:
    """Write table_minimal.tex from the scored minimal pairs."""
    with open(pairs_path, encoding="utf-8") as handle:
        pairs = json.load(handle)
    with open(os.path.join(scores_dir, "scores-seed42.json"), encoding="utf-8") as handle:
        magnitude = np.array(json.load(handle)["magnitude"], dtype=float)
    scales = {
        "Similarity alone": [magnitude],
        r"\texttt{Negation lookup}": [lookup_signed(pairs, magnitude)],
        "Off-the-shelf head": _signed([os.path.join(scores_dir, "scores-offshelf.json")]),
        r"Ours, \textsc{raw}": _signed(glob.glob(os.path.join(scores_dir, "scores-seed*.json"))),
    }
    lexical = glob.glob(os.path.join(scores_dir, "scores-lex-seed*.json"))
    if lexical:
        scales[r"Ours, \textsc{lex}"] = _signed(lexical)
    combined = glob.glob(os.path.join(scores_dir, "scores-fulllex-seed*.json"))
    if combined:
        scales[r"Ours, \textsc{aug+lex}"] = _signed(combined)
    minimal_table(pairs, scales, os.path.join(tex_out, "table_minimal.tex"))
    click.echo(f"{tex_out}/table_minimal.tex")


if __name__ == "__main__":
    main()
