"""The two pictures the composition owes its reader, as pgfplots source.

The article defines a scale on [-100, 100] and, until these figures, never showed it.
Both are drawn from the per-pair dump that ``composition.py --pairs-out`` writes, so
nothing here re-runs a model and nothing can disagree with the numbers in the tables.

**The distribution of the signed score by gold class.** What the composition does, seen
at once: contradictions below zero, entailments high, neutral pairs gathered near the
middle without being pushed negative.

**The reliability of the contradiction probability.** The article claims that fine-tuning
buys calibration rather than the relation. A reliability curve shows it directly: a head
that is calibrated follows the diagonal, one that over-asserts sits above it.

Run::

    PYTHONPATH=src python src/figures_generator/figures_composition.py \\
        --pairs results/v3/pairs-nli-test.json \\
        --zero-shot results/v3/pairs-zs-test.json --tex-out paper/v3
"""

from __future__ import annotations

import json
import os
from typing import Optional

import click
import numpy as np

try:  # PYTHONPATH=src.
    from data.schema import POLARITY_CLASSES
except ImportError:  # pragma: no cover
    from src.data.schema import POLARITY_CLASSES  # type: ignore

#: Okabe-Ito, the same three the rest of the paper uses, validated for colour vision.
COLOURS = {"entailment": "0072B2", "neutral": "999999", "contradiction": "D55E00"}
LABELS = {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"}


def histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of *values* per bin, so classes of different sizes stay comparable."""
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def distribution_figure(pairs: dict, path: str) -> None:
    """Signed score by gold class, as three outlined histograms on one axis."""
    signed = np.array(pairs["signed"], dtype=float)
    truth = np.array(pairs["truth"], dtype=int)
    edges = np.linspace(-100, 100, 41)
    centres = (edges[:-1] + edges[1:]) / 2

    plots = []
    for name, index in POLARITY_CLASSES.items():
        share = histogram(signed[truth == index], edges)
        points = " ".join(f"({x:.1f},{100 * y:.2f})" for x, y in zip(centres, share))
        plots.append(
            f"\\addplot[class{name}] coordinates {{{points}}};\n"
            f"\\addlegendentry{{{LABELS[name]} ($n = {int((truth == index).sum())}$)}}"
        )

    lines = [
        r"% Genere par src/figures_generator/figures_composition.py. Ne pas editer a la main.",
        *[f"\\definecolor{{c{name}}}{{HTML}}{{{code}}}" for name, code in COLOURS.items()],
        r"\begin{figure}[t]",
        r"\centering",
        r"\tikzset{",
        *[f"  class{name}/.style={{draw=c{name}, line width=0.9pt, mark=none, const plot}}," for name in COLOURS],
        r"}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.74\columnwidth, height=4.4cm, scale only axis,",
        r"  axis line style={draw=black!45, line width=0.3pt},",
        r"  axis x line*=bottom, axis y line*=left, y axis line style={draw=none},",
        r"  xmajorgrids=false, ymajorgrids=false, tick align=outside, ytick style={draw=none},",
        r"  xlabel={signed score}, ylabel={share of pairs (\%)},",
        r"  label style={font=\small}, tick label style={font=\small},",
        r"  xmin=-100, xmax=100, ymin=0,",
        r"  legend style={font=\small, draw=none, fill=none, at={(0.02,0.98)},",
        r"    anchor=north west, cells={anchor=west}},",
        r"]",
        r"\draw[draw=black!25, line width=0.3pt, dashed] (axis cs:0,0)"
        r" -- (axis cs:0,\pgfkeysvalueof{/pgfplots/ymax});",
        *plots,
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Signed score by gold class on the SICK half of the test split, "
        r"$\alpha = 2$, binned at width $5$. Contradictions concentrate near $-50$ and "
        r"entailments near $+80$. SICK neutral pairs are related captions rather than "
        r"unrelated sentences, so they belong near the middle of the positive half and "
        r"not at zero; what matters is that the product does not drag them across it.}",
        r"\label{fig:distribution}",
        r"\end{figure}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def reliability(
    probability: np.ndarray, is_contradiction: np.ndarray, bins: int = 10
) -> list[tuple[float, float, int]]:
    """Observed contradiction rate against predicted probability, bin by bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        keep = (probability >= low) & (probability < high if high < 1.0 else probability <= high)
        if keep.sum() == 0:
            continue
        out.append((float(probability[keep].mean()), float(is_contradiction[keep].mean()), int(keep.sum())))
    return out


def reliability_figure(fine_tuned: dict, off_the_shelf: dict, path: str) -> None:
    """Both heads' reliability curves against the diagonal."""
    series = []
    for pairs, style, label in (
        (fine_tuned, "tuned", "fine-tuned"),
        (off_the_shelf, "shelf", "off-the-shelf"),
    ):
        probability = np.array(pairs["p_contradiction"], dtype=float)
        truth = np.array(pairs["truth"], dtype=int)
        curve = reliability(probability, (truth == POLARITY_CLASSES["contradiction"]).astype(float))
        points = " ".join(f"({x:.3f},{y:.3f})" for x, y, _ in curve)
        series.append(f"\\addplot[{style}] coordinates {{{points}}};\n\\addlegendentry{{{label}}}")

    lines = [
        r"% Genere par src/figures_generator/figures_composition.py. Ne pas editer a la main.",
        r"\definecolor{reltuned}{HTML}{0072B2}",
        r"\definecolor{relshelf}{HTML}{E69F00}",
        r"\begin{figure}[t]",
        r"\centering",
        r"\tikzset{",
        r"  tuned/.style={draw=reltuned, line width=0.9pt, mark=*, mark size=1.6pt},",
        r"  shelf/.style={draw=relshelf, line width=0.9pt, mark=square*, mark size=1.5pt},",
        r"}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.78\columnwidth, height=4.4cm, scale only axis,",
        r"  axis line style={draw=black!45, line width=0.3pt},",
        r"  axis x line*=bottom, axis y line*=left, y axis line style={draw=none},",
        r"  xmajorgrids=false, ymajorgrids=false, tick align=outside, ytick style={draw=none},",
        r"  xlabel={predicted $p_{\mathrm{contra}}$}, ylabel={observed rate},",
        r"  label style={font=\small}, tick label style={font=\small},",
        r"  xmin=0, xmax=1, ymin=0, ymax=1,",
        r"  legend style={font=\small, draw=none, fill=none, at={(0.98,0.02)},",
        r"    anchor=south east, cells={anchor=west}},",
        r"]",
        r"\addplot[draw=black!30, line width=0.3pt, dashed, mark=none, forget plot]" + r" coordinates {(0,0) (1,1)};",
        *series,
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Reliability of $p_{\mathrm{contra}}$ on the SICK half of the test split, "
        r"ten equal-width bins. The dashed diagonal is perfect calibration. The "
        r"off-the-shelf head falls far below it: among the pairs to which it gives a "
        r"contradiction probability near $0.5$, fewer than one in ten is a contradiction. "
        r"The fine-tuned head tracks the diagonal.}",
        r"\label{fig:reliability}",
        r"\end{figure}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def composition_table(curve: dict, path: str) -> None:
    """Sensitivity of the composition to the slope, read off the test curve."""
    shown = (1.00, 1.25, 1.50, 1.75, 2.00, 3.00)
    rows = {row["alpha"]: row for row in curve["curve"]}
    lines = [
        r"% Genere par src/figures_generator/figures_composition.py. Ne pas editer a la main.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{l ccc c}",
        r"\toprule",
        r" & Contra. & Entail. & Rel. & Floor \\",
        r"Slope & $<0$ & $>0$ & $r$ & \\",
        r"\midrule",
    ]
    for alpha in shown:
        row = rows.get(alpha)
        if row is None:
            continue
        name = f"$\\alpha = {alpha:.2f}$"
        cells = [
            f"{100 * row['contradictions_negatives']:.1f}",
            f"{100 * row['implications_positives']:.1f}",
            f"{row['pearson_proximite']:.3f}",
            f"${row['plancher']:.0f}$",
        ]
        if alpha == curve["alpha"]:
            name = f"$\\boldsymbol{{\\alpha = {alpha:.2f}}}$"
            cells = [r"\textbf{" + cell.strip("$") + "}" for cell in cells[:3]] + [
                r"$\mathbf{" + f"{row['plancher']:.0f}" + r"}$"
            ]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    only = curve["magnitude_only"]
    lines += [
        r"\addlinespace",
        f"Magnitude alone & {100 * only['contradictions_negatives']:.1f}"
        f" & {100 * only['implications_positives']:.1f} & {only['pearson_proximite']:.3f} & $0$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Composition on the SICK half of the test split, in percent. The slope was "
        r"fitted on development data, which selects $\alpha = 2$ (\textbf{bold}). Slopes below "
        r"$2$ cannot reach $-100$ and are excluded by construction. Last column: the most "
        r"negative score the scale can produce.}",
        r"\label{tab:composition}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


@click.command()
@click.option("--pairs", required=True, help="Per-pair dump for the fine-tuned head.")
@click.option("--curve", "curve_path", default=None, help="Composition curve of the fine-tuned head.")
@click.option("--zero-shot", "zero_shot", default=None, help="Same, for the off-the-shelf head.")
@click.option("--tex-out", default="paper/v3", show_default=True)
def main(pairs: str, curve_path: Optional[str], zero_shot: Optional[str], tex_out: str) -> None:
    """Write both figures as pgfplots source."""
    with open(pairs, encoding="utf-8") as handle:
        tuned = json.load(handle)
    os.makedirs(tex_out, exist_ok=True)

    distribution_figure(tuned, os.path.join(tex_out, "figure_distribution.tex"))
    click.echo(f"distribution : {tex_out}/figure_distribution.tex")

    if curve_path:
        with open(curve_path, encoding="utf-8") as handle:
            composition_table(json.load(handle), os.path.join(tex_out, "table_composition.tex"))
        click.echo(f"composition  : {tex_out}/table_composition.tex")

    if zero_shot:
        with open(zero_shot, encoding="utf-8") as handle:
            shelf = json.load(handle)
        reliability_figure(tuned, shelf, os.path.join(tex_out, "figure_reliability.tex"))
        click.echo(f"fiabilite    : {tex_out}/figure_reliability.tex")


if __name__ == "__main__":
    main()
