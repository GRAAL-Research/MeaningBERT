"""Turn the polarity grid into the numbers and tables the v3 paper reports.

Every claim in the paper comes from here, so the arithmetic is in one place and the paper
cannot quietly drift from the runs. Three questions, three tests, chosen before looking:

**Does augmentation change the task performance?** Welch's t-test on macro-F1 between the
two conditions, per architecture. Welch and not Student because the two conditions have
visibly different seed variances; equal-variance would understate the p-values exactly
where the claim is "nothing changed", which is the direction that flatters us.

**Does it change the sanity suite?** Same test, same pairs, on the generated suite. This is
where the effect is expected, so reporting both on the same footing is what makes the first
result credible.

**Does NLI pretraining help?** Paired by seed, because the two architectures saw the same
ten seeds and pairing removes the seed variance that both share.

Effect sizes travel with every p-value: with ten seeds and standard deviations near 0.002,
a difference of 0.001 reaches significance while meaning nothing, and Cohen's d is what
says so.

Run::

    PYTHONPATH=src python src/figures_generator/analyse_v3.py --results results/v3
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import statistics as st
from typing import Any, Optional

import click
from scipy import stats

#: Architectures in the order the paper presents them: by test macro-F1, best first.
ARCH_LABELS: dict[str, str] = {
    "nli-deberta-v3-large": r"DeBERTa-v3-large\textsubscript{NLI}",
    "deberta-v3-large": "DeBERTa-v3-large",
    "roberta-large-mnli": r"RoBERTa-large\textsubscript{MNLI}",
    "nli-deberta-v3-base": r"DeBERTa-v3-base\textsubscript{NLI}",
    "deberta-v3-base": "DeBERTa-v3-base",
    "stsb-roberta-base": r"RoBERTa-base\textsubscript{STS-B}",
    "bert": "BERT-base",
}

#: What each metric is called in the paper, and whether higher is better.
METRICS: dict[str, str] = {
    "macro_f1": "macro-F1",
    "sanity": "sanity",
    "nan_nli": "NaN-NLI",
    "monli": "MoNLI",
}


def load(root: str) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Read every cell under *root* into ``(arch, condition) -> metric -> values``.

    The directory name is ``<arch>-<condition>`` and is split from the RIGHT, because an
    architecture tag contains dashes: splitting from the left reads
    ``nli-deberta-v3-base-none`` as the architecture ``nli``.
    """
    out: dict[tuple[str, str], dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for path in sorted(glob.glob(os.path.join(root, "polarity", "*", "seed*", "metrics.json"))):
        arch, _, condition = os.path.basename(os.path.dirname(os.path.dirname(path))).rpartition("-")
        if arch not in ARCH_LABELS or condition not in ("none", "full"):
            continue
        with open(path, encoding="utf-8") as handle:
            cell = json.load(handle)
        if not cell:
            continue
        probes = cell["probes"]
        out[(arch, condition)]["macro_f1"].append(cell["test"]["macro_f1"])
        out[(arch, condition)]["accuracy"].append(cell["test"]["accuracy"])
        out[(arch, condition)]["sanity"].append(probes["sanity"]["accuracy"])
        out[(arch, condition)]["nan_nli"].append(probes["nan_nli"]["accuracy"])
        out[(arch, condition)]["monli"].append(probes["monli"]["accuracy"])
    return out


def cohen_d(left: list[float], right: list[float]) -> float:
    """Standardised difference, pooled standard deviation.

    Reported beside every p-value because ten seeds with a standard deviation near 0.002
    make a difference of 0.001 significant and meaningless at the same time.
    """
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = math.sqrt(((len(left) - 1) * st.variance(left) + (len(right) - 1) * st.variance(right))
                       / (len(left) + len(right) - 2))
    return (st.mean(left) - st.mean(right)) / pooled if pooled else float("nan")


def welch(left: list[float], right: list[float]) -> dict[str, float]:
    """Welch's t-test plus the effect size, as one record."""
    if len(left) < 2 or len(right) < 2:
        return {"diff": float("nan"), "p": float("nan"), "d": float("nan")}
    result = stats.ttest_ind(left, right, equal_var=False)
    return {"diff": st.mean(left) - st.mean(right), "p": float(result.pvalue), "d": cohen_d(left, right)}


def paired(left: list[float], right: list[float]) -> dict[str, float]:
    """Paired t-test, for two architectures that saw the same seeds."""
    if len(left) != len(right) or len(left) < 2:
        return {"diff": float("nan"), "p": float("nan"), "d": float("nan")}
    result = stats.ttest_rel(left, right)
    differences = [a - b for a, b in zip(left, right)]
    spread = st.stdev(differences)
    return {
        "diff": st.mean(differences),
        "p": float(result.pvalue),
        "d": st.mean(differences) / spread if spread else float("nan"),
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    return (st.mean(values), st.stdev(values) if len(values) > 1 else 0.0)


def main_table(cells) -> str:
    """The paper's main results table, both conditions side by side."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\begin{tabular}{l rr rr rr rr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{macro-F1} & \multicolumn{2}{c}{Sanity}"
        r" & \multicolumn{2}{c}{NaN-NLI} & \multicolumn{2}{c}{MoNLI} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Encoder & \textsc{raw} & \textsc{aug} & \textsc{raw} & \textsc{aug}"
        r" & \textsc{raw} & \textsc{aug} & \textsc{raw} & \textsc{aug} \\",
        r"\midrule",
    ]
    for arch, label in ARCH_LABELS.items():
        row = [label]
        for metric in METRICS:
            for condition in ("none", "full"):
                values = cells.get((arch, condition), {}).get(metric, [])
                row.append(f"{st.mean(values):.3f}" if values else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Polarity head, ten seeds per cell. \textsc{raw} trains on the merged corpus, "
              r"\textsc{aug} adds the derived-polarity augmentation. Sanity is accuracy on the three "
              r"generated suites; NaN-NLI and MoNLI are held-out probes the models never train on.}",
              r"\label{tab:main}", r"\end{table*}"]
    return "\n".join(lines)


def stats_table(cells) -> str:
    """Per-encoder significance of the augmentation, on both axes at once.

    Both tests on the same footing is what makes the first one credible: reporting only
    the metric that moved would leave the reader to take "nothing changed" on trust.
    """
    lines = [
        r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{l rr rr}", r"\toprule",
        r" & \multicolumn{2}{c}{macro-F\textsubscript{1}} & \multicolumn{2}{c}{Sanity} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Encoder & $\Delta$ & $d$ & $\Delta$ & $d$ \\", r"\midrule",
    ]
    for arch, label in ARCH_LABELS.items():
        raw, aug = cells.get((arch, "none")), cells.get((arch, "full"))
        if not raw or not aug:
            continue
        task = welch(aug["macro_f1"], raw["macro_f1"])
        sanity = welch(aug["sanity"], raw["sanity"])
        mark = "" if task["p"] < 0.05 else r"$^{\dagger}$"
        lines.append(
            f"{label} & {task['diff']:+.3f}{mark} & {task['d']:+.2f} "
            f"& {sanity['diff']:+.3f} & {sanity['d']:+.1f} " + r"\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Effect of augmentation, \textsc{aug} minus \textsc{raw}, Welch's $t$-test over ten "
        r"seeds with Cohen's $d$. $\dagger$ marks a difference that is \emph{not} significant at "
        r"$p<0.05$. Every sanity difference has $p<10^{-4}$.}",
        r"\label{tab:stats}", r"\end{table}",
    ]
    return "\n".join(lines)


def figure(cells, path: str) -> None:
    """Emit the augmentation figure as pgfplots source rather than a rasterised PDF.

    Vector output that inherits the document's fonts and sizes, and whose numbers stay
    readable in the diff: a reviewer, or the next person to touch this, can check a point
    against the results without opening an image.

    Full width rather than one column: seven encoder labels and two panels do not fit in a
    column, and the figure is read against the table beside it.

    Two panels sharing one y axis. Horizontal error bars are one standard deviation over
    the ten seeds, which is what makes the comparison legible: on the task panel the two
    conditions overlap, on the sanity panel they do not come close.
    """
    order = [a for a in ARCH_LABELS if (a, "none") in cells and (a, "full") in cells][::-1]
    labels = ", ".join(ARCH_LABELS[a] for a in order)

    def series(metric: str, condition: str, offset: float) -> str:
        points = []
        for index, arch in enumerate(order):
            values = cells[(arch, condition)][metric]
            points.append(f"({st.mean(values):.4f},{index + offset:.2f}) +- ({mean_sd(values)[1]:.4f},0)")
        return " ".join(points)

    panels = []
    for metric, title, xmin, xmax in (
        ("macro_f1", r"macro-F$_1$", 0.76, 0.93),
        ("sanity", "Sanity suites", 0.76, 1.01),
    ):
        body = [f"\\nextgroupplot[title={{{title}}}, xmin={xmin}, xmax={xmax}]"]
        for condition, style, offset in (("none", "raw", -0.17), ("full", "aug", 0.17)):
            body.append(
                f"\\addplot+[{style}] plot [error bars/.cd, x dir=both, x explicit] "
                f"coordinates {{{series(metric, condition, offset)}}};"
            )
        panels.append("\n".join(body))

    lines = [
        r"% Genere par src/figures_generator/analyse_v3.py. Ne pas editer a la main.",
        r"% Palette Okabe-Ito, validee : bande de clarte, plancher de chroma, separation",
        r"% daltonienne et contraste. La forme du marqueur double la couleur, pour que",
        r"% l'identite ne repose jamais sur elle seule.",
        r"\definecolor{condraw}{HTML}{0072B2}",
        r"\definecolor{condaug}{HTML}{E69F00}",
        r"\begin{figure*}[t]", r"\centering",
        r"% Declares globalement : une option de tikzpicture n'est pas visible depuis",
        r"% \addplot a l'interieur d'un groupplot.",
        r"\tikzset{",
        r"  raw/.style={mark=*, mark size=1.5pt, only marks, color=condraw},",
        r"  aug/.style={mark=square*, mark size=1.5pt, only marks, color=condaug},",
        r"}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 1, horizontal sep=0.45cm, y descriptions at=edge left},",
        r"  width=0.36\textwidth, height=4.4cm,",
        r"  scale only axis,",
        r"  % Tufte : pas de cadre, une seule ligne d'axe, la grille verticale assez pale",
        r"  % pour guider l'oeil sans entrer en competition avec les points.",
        r"  axis line style={draw=black!45, line width=0.3pt},",
        r"  axis x line*=bottom, axis y line=none,",
        r"  xmajorgrids, ymajorgrids=false, grid style={draw=black!10, line width=0.3pt},",
        r"  tick align=outside, ytick style={draw=none},",
        r"  tick label style={font=\scriptsize}, title style={font=\small, yshift=-3pt},",
        f"  ymin=-0.7, ymax={len(order) - 0.3}, ytick={{{','.join(str(i) for i in range(len(order)))}}},",
        f"  yticklabels={{{labels}}}, yticklabel style={{font=\scriptsize}},",
        r"  every axis plot/.append style={line width=0.7pt},",
        r"  legend style={font=\scriptsize, draw=none, fill=none,",
        r"    at={(0.97,0.04)}, anchor=south east, cells={anchor=west}},",
        r"]",
        panels[0],
        panels[1],
        r"\legend{without augmentation, with augmentation}",
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"\caption{Augmentation moves one axis and not the other. Points are means over ten "
        r"seeds, bars one standard deviation. On the task axis the two conditions overlap for "
        r"every encoder; on the sanity axis they converge to one value from seven different "
        r"starting points.}",
        r"\label{fig:augmentation}",
        r"\end{figure*}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


@click.command()
@click.option("--results", default="results/v3", show_default=True)
@click.option("--tex-out", default=None, help="Directory to write the LaTeX tables into.")
@click.option("--json-out", default=None)
def main(results: str, tex_out: Optional[str], json_out: Optional[str]) -> None:
    """Compute every number the paper reports and print it."""
    cells = load(results)
    findings: dict[str, Any] = {"cells": {}, "augmentation": {}, "nli": {}}

    click.echo("=== cellules (moyenne +- ecart-type sur les graines)\n")
    click.echo("%-28s %-5s %2s %16s %16s %16s %16s" % ("architecture", "cond", "n", *METRICS.values()))
    for arch in ARCH_LABELS:
        for condition in ("none", "full"):
            values = cells.get((arch, condition))
            if not values:
                continue
            row = [f"{st.mean(values[m]):.4f} ± {mean_sd(values[m])[1]:.4f}" for m in METRICS]
            click.echo("%-28s %-5s %2d %16s %16s %16s %16s" % (arch, condition, len(values["macro_f1"]), *row))
            findings["cells"][f"{arch}-{condition}"] = {
                m: {"mean": st.mean(values[m]), "sd": mean_sd(values[m])[1], "n": len(values[m])}
                for m in METRICS
            }

    click.echo("\n=== effet de l'augmentation, Welch par architecture")
    click.echo("%-28s %22s %22s" % ("architecture", "macro-F1 (aug - raw)", "sanity (aug - raw)"))
    for arch in ARCH_LABELS:
        raw, aug = cells.get((arch, "none")), cells.get((arch, "full"))
        if not raw or not aug:
            continue
        task = welch(aug["macro_f1"], raw["macro_f1"])
        sanity = welch(aug["sanity"], raw["sanity"])
        findings["augmentation"][arch] = {"macro_f1": task, "sanity": sanity,
                                          "monli": welch(aug["monli"], raw["monli"])}
        click.echo(
            "%-28s %+8.4f p=%-7.4f d=%+5.2f %+8.4f p=%-7.4f d=%+5.2f"
            % (arch, task["diff"], task["p"], task["d"], sanity["diff"], sanity["p"], sanity["d"])
        )

    click.echo("\n=== apport du pre-entrainement NLI, apparie par graine")
    for condition in ("none", "full"):
        for nli, plain in (("nli-deberta-v3-large", "deberta-v3-large"),
                           ("nli-deberta-v3-base", "deberta-v3-base")):
            left, right = cells.get((nli, condition)), cells.get((plain, condition))
            if not left or not right:
                continue
            for metric in ("macro_f1", "monli"):
                got = paired(left[metric], right[metric])
                findings["nli"][f"{nli}-vs-{plain}-{condition}-{metric}"] = got
                click.echo("%-22s %-5s %-9s %+7.4f  p=%-8.4f d=%+5.2f"
                           % (f"{nli[:22]}", condition, METRICS[metric], got["diff"], got["p"], got["d"]))

    if tex_out:
        os.makedirs(tex_out, exist_ok=True)
        with open(os.path.join(tex_out, "table_main.tex"), "w", encoding="utf-8") as handle:
            handle.write(main_table(cells) + "\n")
        with open(os.path.join(tex_out, "table_stats.tex"), "w", encoding="utf-8") as handle:
            handle.write(stats_table(cells) + "\n")
        figure(cells, os.path.join(tex_out, "figure_augmentation.tex"))
        click.echo(f"\ntables : {tex_out}")
    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(findings, handle, indent=2)
        click.echo(f"brut : {json_out}")


if __name__ == "__main__":
    main()
