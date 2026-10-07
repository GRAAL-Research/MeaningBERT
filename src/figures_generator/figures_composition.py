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
import math
import os
from typing import Optional, Union

import click
import numpy as np

try:  # PYTHONPATH=src.
    from data.schema import POLARITY_CLASSES
except ImportError:  # pragma: no cover
    from src.data.schema import POLARITY_CLASSES  # type: ignore

#: Okabe-Ito, the same three the rest of the paper uses, validated for colour vision.
COLOURS = {"entailment": "0072B2", "neutral": "009E73", "contradiction": "D55E00"}
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
        plots.append(f"\\addplot[draw=c{name}, line width=0.9pt, mark=none, const plot] " f"coordinates {{{points}}};")

    # La legende vit dans la legende de figure, en couleur. Dans le panneau, trois
    # entrees et leurs effectifs occupaient le quart de la surface utile et couvraient
    # le pic des contradictions.
    named = ", ".join(f"\\textcolor{{c{name}}}{{\\textbf{{{LABELS[name]}}}}}" for name in POLARITY_CLASSES)

    lines = [
        r"% Genere par src/figures_generator/figures_composition.py. Ne pas editer a la main.",
        *[f"\\definecolor{{c{name}}}{{HTML}}{{{code}}}" for name, code in COLOURS.items()],
        r"\begin{figure}[t]",
        r"\centering",
        r"% Options en clair plutot qu'un style nomme : un style \tikzset vit sous",
        r"% /tikz/ et \addplot resout sous /pgfplots/, ce qui casse selon la version.",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.74\columnwidth, height=3.7cm, scale only axis,",
        r"  axis line style={draw=black!45, line width=0.3pt},",
        r"  axis x line*=bottom, axis y line*=left, y axis line style={draw=none},",
        r"  xmajorgrids=false, ymajorgrids=false, tick align=outside, ytick style={draw=none},",
        r"  xlabel={Signed score}, ylabel={Share of pairs (\%)},",
        r"  label style={font=\small}, tick label style={font=\small},",
        r"  xmin=-100, xmax=100, ymin=0,",
        r"]",
        r"\draw[draw=black!25, line width=0.3pt, dashed] (axis cs:0,0)"
        r" -- (axis cs:0,\pgfkeysvalueof{/pgfplots/ymax});",
        *plots,
        r"\end{axis}",
        r"\end{tikzpicture}",
        f"\\caption{{Signed score by gold class (\\autoref{{eq:compose}}, one polarity head), bins "
        f"of width $5$: {named}. Dashed line: zero.}}",
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


def mark_style(colour: str, mark: str, size: str) -> str:
    """Plot options written out rather than hidden behind a named style.

    A style declared with ``\\tikzset`` lives under ``/tikz/`` while ``\\addplot``
    resolves its options under ``/pgfplots/``. Whether the fallback happens depends on
    the pgfplots version, and where it does not the key is unknown, the path aborts, and
    TikZ reports "Giving up on this path" on the *following* line. Writing the options
    out removes the lookup, and with it the version dependence.
    """
    fill = f"mark options={{draw={colour}, fill={colour}}}"
    return f"draw={colour}, line width=0.9pt, mark={mark}, mark size={size}, {fill}"


def reliability_figure(fine_tuned: dict, off_the_shelf: dict, path: str) -> None:
    """Both heads' reliability curves against the diagonal."""
    tuned = mark_style("reltuned", "*", "1.6pt")
    shelf = mark_style("relshelf", "square*", "1.5pt")
    series = []
    for pairs, style, _ in ((fine_tuned, tuned, "fine-tuned"), (off_the_shelf, shelf, "off-the-shelf")):
        probability = np.array(pairs["p_contradiction"], dtype=float)
        truth = np.array(pairs["truth"], dtype=int)
        curve = reliability(probability, (truth == POLARITY_CLASSES["contradiction"]).astype(float))
        points = " ".join(f"({x:.3f},{y:.3f})" for x, y, _ in curve)
        series.append(f"\\addplot[{style}] coordinates {{{points}}};")

    lines = [
        r"% Genere par src/figures_generator/figures_composition.py. Ne pas editer a la main.",
        r"\definecolor{reltuned}{HTML}{0072B2}",
        r"\definecolor{relshelf}{HTML}{D55E00}",
        r"\begin{figure}[t]",
        r"\centering",
        r"% Options en clair plutot qu'un style nomme, pour la meme raison.",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.78\columnwidth, height=3.7cm, scale only axis,",
        r"  axis line style={draw=black!45, line width=0.3pt},",
        r"  axis x line*=bottom, axis y line*=left, y axis line style={draw=none},",
        r"  xmajorgrids=false, ymajorgrids=false, tick align=outside, ytick style={draw=none},",
        r"  xlabel={Predicted $p_{\mathrm{contra}}$}, ylabel={Observed rate},",
        r"  label style={font=\small}, tick label style={font=\small},",
        r"  xmin=0, xmax=1, ymin=0, ymax=1,",
        r"]",
        r"\addplot[draw=black!30, line width=0.3pt, dashed, mark=none, forget plot]" + r" coordinates {(0,0) (1,1)};",
        *series,
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Observed contradiction rate against predicted $p_{\mathrm{contra}}$, SICK test half, "
        r"ten bins, for the \textcolor{reltuned}{\textbf{fine-tuned head}} and the "
        r"\textcolor{relshelf}{\textbf{off-the-shelf head}}. Dashed diagonal: perfect calibration.}",
        r"\label{fig:reliability}",
        r"\end{figure}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


#: Head counts written out, as the paper writes "ten random seeds".
HEAD_WORDS = {10: "ten"}


def mean_sd(values: list[float], decimals: int = 2) -> str:
    """``mean`` alone for one value, ``mean`` with the standard deviation as subscript otherwise."""
    if len(values) == 1:
        return f"{values[0]:.{decimals}f}"
    return f"{np.mean(values):.{decimals}f}$_{{\\pm {np.std(values, ddof=1):.{decimals}f}}}$"


def composition_table(curves: Union[dict, list[dict]], path: str) -> None:
    """Sensitivity of the composition to the factor, read off the test curves.

    One curve per polarity head; with several, each cell is a mean and a standard
    deviation over the heads, since a single checkpoint would let one seed carry the
    section. The neutral column is here because the paper's title is about that class: a
    scale that fixed contradictions by dragging unrelated pairs negative would satisfy the
    two sign criteria and fail the thing the product form exists for.
    """
    curves = [curves] if isinstance(curves, dict) else curves
    curve = curves[0]
    shown = (1.00, 1.25, 1.50, 1.75, 2.00, 3.00)
    per_head = [{row["alpha"]: row for row in each["curve"]} for each in curves]
    rows = per_head[0]
    lines = [
        r"% Genere par src/figures_generator/figures_composition.py. Ne pas editer a la main.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{l ccc c}",
        r"\toprule",
        r" & Contra. & Neutral & Rel. & Floor \\",
        r"Factor & $<0$ & $>0$ & $r$ & \\",
        r"\midrule",
    ]
    for alpha in shown:
        row = rows.get(alpha)
        if row is None:
            continue
        name = f"$\\alpha = {alpha:.2f}$"
        heads = [head[alpha] for head in per_head if alpha in head]
        cells = [
            mean_sd([100 * head["contradictions_negatives"] for head in heads]),
            mean_sd([100 * head["neutres_positifs"] for head in heads]),
            mean_sd([head["pearson_proximite"] for head in heads]),
        ]
        floor = f"${row['plancher']:.0f}$"
        if alpha == curve["alpha"]:
            name = f"$\\boldsymbol{{\\alpha = {alpha:.2f}}}$"
            cells = [r"\textbf{" + cell + "}" for cell in cells]
            floor = r"$\mathbf{" + f"{row['plancher']:.0f}" + r"}$"
        lines.append(f"{name} & " + " & ".join(cells + [floor]) + r" \\")
    only = curve["magnitude_only"]
    lines += [
        f"Magnitude alone & {100 * only['contradictions_negatives']:.2f}"
        f" & {100 * only['neutres_positifs']:.2f}"
        f" & {only['pearson_proximite']:.2f} & $0$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Composed score per factor $\alpha$ in place of the $2$ of \autoref{eq:compose}, "
        r"SICK test half (1{,}404 entailments, 2{,}000 "
        r"neutral, 712 contradictions), in percent"
        + (
            ", mean with standard deviation as subscript over "
            f"{HEAD_WORDS.get(len(curves), len(curves))} polarity heads"
            if len(curves) > 1
            else ""
        )
        + r". \textbf{Bold}: the factor of \autoref{eq:compose}.}",
        r"\label{tab:composition}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def accepted_rates(score: np.ndarray, truth: np.ndarray, cut: float) -> tuple[float, float]:
    """Share of accepted pairs that are contradictions, and share of entailments kept."""
    accepted = score > cut
    entailments = (truth == POLARITY_CLASSES["entailment"]).sum()
    share = float((truth[accepted] == POLARITY_CLASSES["contradiction"]).mean()) if accepted.any() else float("nan")
    recall = (
        float((accepted & (truth == POLARITY_CLASSES["entailment"])).sum() / entailments)
        if entailments
        else float("nan")
    )
    return share, recall


def bootstrap_interval(
    score: np.ndarray, truth: np.ndarray, cut: float, draws: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """Percentile interval on the contradiction share, resampling pairs with replacement.

    The share is what the paper's contribution now rests on, so it needs an uncertainty
    rather than a reader's willingness to assume that 712 contradictions are enough.
    """
    rng = np.random.default_rng(seed)
    shares = []
    for row in rng.integers(0, len(truth), size=(draws, len(truth))):
        value, _ = accepted_rates(score[row], truth[row], cut)
        shares.append(value)
    return tuple(np.nanpercentile(shares, [2.5, 97.5]))


def decision_point(scales: dict, truth: np.ndarray, thresholds=(25, 50, 70)) -> dict:
    """What each scale accepts as preserved meaning, at the same cut.

    The article's claim until now was that the signed scale expresses a distinction the
    magnitude cannot, which is a statement about expressiveness and not about being a
    better metric. This is the measurement that makes it one: at a fixed acceptance
    threshold, how many of the pairs waved through are in fact contradictions, judged by
    SICK's own inference labels rather than by our conversion rule.

    The entailment column is the control. A scale that simply shifts everything downwards
    would also accept fewer contradictions, and would pay for it by rejecting entailments.
    """
    out = {}
    for name, score in scales.items():
        rows = []
        for cut in thresholds:
            share, recall = accepted_rates(score, truth, cut)
            low, high = bootstrap_interval(score, truth, cut)
            rows.append({"threshold": cut, "share": share, "recall": recall, "low": low, "high": high})
        out[name] = rows
    return out


def decision_table(
    tuned: dict,
    off_the_shelf: dict,
    path: str,
    thresholds=(25, 50, 70),
    heads: Optional[list[dict]] = None,
    lookup: Optional[list[bool]] = None,
    lexical: Optional[list[dict]] = None,
    combined: Optional[list[dict]] = None,
) -> None:
    """The decision-point comparison, one row per scale and two blocks.

    ``heads`` holds the per-pair dumps of every fine-tuned head and ``lexical`` those of
    the heads trained with lexical contradictions; each group reports a mean and a
    standard deviation over its heads.
    """
    truth = np.array(tuned["truth"], dtype=int)
    scales = {
        "Magnitude alone": np.array(tuned["magnitude"], dtype=float),
        **(
            {
                "Negation lookup": np.where(np.array(lookup, dtype=bool), -1.0, 1.0)
                * np.array(tuned["magnitude"], dtype=float)
            }
            if lookup is not None
            else {}
        ),
        "Off-the-shelf head": np.array(off_the_shelf["signed"], dtype=float),
        "Ours, \\textsc{raw}": np.array(tuned["signed"], dtype=float),
        **({"Ours, \\textsc{lex}": np.array(lexical[0]["signed"], dtype=float)} if lexical else {}),
        **({"Ours, \\textsc{aug+lex}": np.array(combined[0]["signed"], dtype=float)} if combined else {}),
    }
    got = decision_point(scales, truth, thresholds)
    widest = max((row["high"] - row["low"]) / 2 for rows in got.values() for row in rows if not math.isnan(row["low"]))
    header = " & ".join(f"$s > {cut}$" for cut in thresholds)

    lines = [
        r"% Genere par src/figures_generator/figures_composition.py. Ne pas editer a la main.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{l " + "c" * len(thresholds) + "}",
        r"\toprule",
        f"Scale & {header} " + r"\\",
        r"\midrule",
        r"\multicolumn{" + str(len(thresholds) + 1) + r"}{l}{\emph{Contradictions accepted}} \\",
    ]
    groups = {
        "Ours, \\textsc{raw}": heads or [],
        "Ours, \\textsc{lex}": lexical or [],
        "Ours, \\textsc{aug+lex}": combined or [],
    }

    def spread(group: list[dict], key: str) -> list[list[float]]:
        index = 0 if key == "share" else 1
        return [
            [
                100 * accepted_rates(np.array(h["signed"], dtype=float), np.array(h["truth"], dtype=int), cut)[index]
                for h in group
            ]
            for cut in thresholds
        ]

    def cells_for(name: str, rows: list, key: str) -> list[str]:
        if len(groups.get(name, [])) > 1:
            return [mean_sd(values) for values in spread(groups[name], key)]
        return [f"{100 * row[key]:.2f}" for row in rows]

    for name, rows in got.items():
        lines.append(f"\\quad {name} & " + " & ".join(cells_for(name, rows, "share")) + r" \\")
    lines += [
        r"\multicolumn{" + str(len(thresholds) + 1) + r"}{l}{\emph{Entailments kept}} \\",
    ]
    for name, rows in got.items():
        lines.append(f"\\quad {name} & " + " & ".join(cells_for(name, rows, "recall")) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Contradictions accepted and entailments kept by each scale at three thresholds, "
        r"in percent. Percentile bootstrap intervals (1{,}000 resamplings) are at most "
        f"$\\pm{100 * widest:.1f}$ wide"
        + (
            "; ours: mean with standard deviation as subscript over " f"{HEAD_WORDS.get(len(heads), len(heads))} heads"
            if heads and len(heads) > 1
            else ""
        )
        + ".}",
        r"\label{tab:decision}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


@click.command()
@click.option("--pairs", required=True, help="Per-pair dump for the fine-tuned head.")
@click.option("--curve", "curve_path", default=None, help="Composition curve of the fine-tuned head.")
@click.option("--zero-shot", "zero_shot", default=None, help="Same, for the off-the-shelf head.")
@click.option("--seed-curve", "seed_curves", multiple=True, help="Curve of a further fine-tuned head.")
@click.option("--seed-pairs", "seed_pairs", multiple=True, help="Per-pair dump of a further fine-tuned head.")
@click.option("--lookup", "lookup_path", default=None, help="Negation-lookup flags for the same pairs.")
@click.option("--lex-pairs", "lex_pairs", multiple=True, help="Per-pair dump of a head trained with lexical pairs.")
@click.option(
    "--aug-lex-pairs", "aug_lex_pairs", multiple=True, help="Same, for heads trained with both augmentations."
)
@click.option("--tex-out", default="paper/v3", show_default=True)
def main(
    pairs: str,
    curve_path: Optional[str],
    zero_shot: Optional[str],
    seed_curves: tuple[str, ...],
    seed_pairs: tuple[str, ...],
    lookup_path: Optional[str],
    lex_pairs: tuple[str, ...],
    aug_lex_pairs: tuple[str, ...],
    tex_out: str,
) -> None:
    """Write both figures as pgfplots source."""
    with open(pairs, encoding="utf-8") as handle:
        tuned = json.load(handle)
    os.makedirs(tex_out, exist_ok=True)

    distribution_figure(tuned, os.path.join(tex_out, "figure_distribution.tex"))
    click.echo(f"distribution : {tex_out}/figure_distribution.tex")

    if curve_path:
        with open(curve_path, encoding="utf-8") as handle:
            curves = [json.load(handle)]
        for extra in seed_curves:
            with open(extra, encoding="utf-8") as handle:
                curves.append(json.load(handle))
        composition_table(curves, os.path.join(tex_out, "table_composition.tex"))
        click.echo(f"composition  : {tex_out}/table_composition.tex")

    if zero_shot:
        with open(zero_shot, encoding="utf-8") as handle:
            shelf = json.load(handle)
        reliability_figure(tuned, shelf, os.path.join(tex_out, "figure_reliability.tex"))
        heads = [tuned]
        for extra in seed_pairs:
            with open(extra, encoding="utf-8") as handle:
                heads.append(json.load(handle))
        lookup = None
        if lookup_path:
            with open(lookup_path, encoding="utf-8") as handle:
                lookup = json.load(handle)["flags"]
        lexical = []
        for extra in lex_pairs:
            with open(extra, encoding="utf-8") as handle:
                lexical.append(json.load(handle))
        combined = []
        for extra in aug_lex_pairs:
            with open(extra, encoding="utf-8") as handle:
                combined.append(json.load(handle))
        decision_table(
            tuned,
            shelf,
            os.path.join(tex_out, "table_decision.tex"),
            heads=heads,
            lookup=lookup,
            lexical=lexical,
            combined=combined,
        )
        click.echo(f"point de decision : {tex_out}/table_decision.tex")
        click.echo(f"fiabilite    : {tex_out}/figure_reliability.tex")


if __name__ == "__main__":
    main()
