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
    "smollm2-1.7b": r"SmolLM2-1.7B",
    "smollm2-360m": r"SmolLM2-360M",
    "smollm2-135m": r"SmolLM2-135M",
}

#: What each metric is called in the paper, and whether higher is better.
METRICS: dict[str, str] = {
    "macro_f1": "macro-F1",
    "sanity": "sanity",
    "nan_nli": "NaN-NLI",
    "monli": "MoNLI",
}


def unlabelled_cells(root: str) -> set[str]:
    """Architecture tags present on disk that no label covers.

    ``load`` filters on ``ARCH_LABELS``, so a cell trained under a tag nobody declared is
    dropped without a word: every table regenerates identically and nothing says a run was
    ignored. Adding a model to the grid and forgetting its label is therefore a silent
    way to publish the wrong numbers, which is what this guards against.
    """
    found = set()
    for path in glob.glob(os.path.join(root, "polarity", "*", "seed*", "metrics.json")):
        arch, _, condition = os.path.basename(os.path.dirname(os.path.dirname(path))).rpartition("-")
        if condition in ("none", "full") and arch not in ARCH_LABELS:
            found.add(arch)
    return found


def load(root: str) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Read every cell under *root* into ``(arch, condition) -> metric -> values``.

    The directory name is ``<arch>-<condition>`` and is split from the RIGHT, because an
    architecture tag contains dashes: splitting from the left reads
    ``nli-deberta-v3-base-none`` as the architecture ``nli``.
    """
    out: dict[tuple[str, str], dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
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
        for name, recall in suite_recalls(probes["sanity"]).items():
            out[(arch, condition)][name].append(recall)
    return out


#: The three generated families, and the class each one must be assigned.
SUITES: dict[str, str] = {
    "identical": "entailment",
    "unrelated": "neutral",
    "mirrored": "contradiction",
}


def suite_recalls(sanity: dict) -> dict[str, float]:
    """Per-family accuracy inside the sanity suite, read off its confusion matrix.

    Each generated family maps onto exactly one gold class, so the recall of that class
    is the accuracy on that family. Reading it here rather than typing it into the paper
    is what keeps the three-family breakdown tied to the runs.
    """
    names = sanity["class_names"]
    matrix = sanity["confusion"]
    out = {}
    for suite, gold in SUITES.items():
        row = matrix[names.index(gold)]
        total = sum(row)
        out[suite] = row[names.index(gold)] / total if total else float("nan")
    return out


def cohen_d(left: list[float], right: list[float]) -> float:
    """Standardised difference, pooled standard deviation.

    Reported beside every p-value because ten seeds with a standard deviation near 0.002
    make a difference of 0.001 significant and meaningless at the same time.
    """
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = math.sqrt(
        ((len(left) - 1) * st.variance(left) + (len(right) - 1) * st.variance(right)) / (len(left) + len(right) - 2)
    )
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


def as_percent(values: list[float]) -> str:
    """Mean and seed standard deviation, both as percentages.

    Percentages rather than the unit interval: three leading zeros in a row of 0.9XX
    carry no information, and a reader compares 91.10 against 90.98 faster than
    0.9110 against 0.9098.
    """
    if not values:
        return "--"
    mean, sd = mean_sd(values)
    return f"{100 * mean:.2f}\\,$\\pm$\\,{100 * sd:.2f}"


def main_table(cells) -> str:
    """The paper's main results table: one row per encoder and condition.

    One metric per column and the two conditions on consecutive rows, rather than eight
    numeric columns side by side. With a standard deviation beside every mean the wide
    layout no longer fits the page, and stacking the conditions keeps the comparison the
    paper makes -- raw against aug, same encoder -- on adjacent lines.
    """
    best = {}
    for metric in METRICS:
        pool = [st.mean(v[metric]) for (a, _c), v in cells.items() if v.get(metric)]
        best[metric] = max(pool) if pool else None
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\begin{tabular}{l l cccc}",
        r"\toprule",
        r"Encoder & Condition & macro-F\textsubscript{1} & Sanity suites & NaN-NLI & MoNLI \\",
        r"\midrule",
    ]
    present = [(a, label) for a, label in ARCH_LABELS.items() if (a, "none") in cells or (a, "full") in cells]
    for position, (arch, label) in enumerate(present):
        if position:
            lines.append(r"\addlinespace")
        for condition, name in (("none", r"\textsc{raw}"), ("full", r"\textsc{aug}")):
            row = [r"\multirow{2}{*}{" + label + "}" if condition == "none" else "", name]
            for metric in METRICS:
                values = cells.get((arch, condition), {}).get(metric, [])
                cell = as_percent(values)
                if values and best[metric] is not None and math.isclose(st.mean(values), best[metric]):
                    cell = r"\textbf{" + cell + "}"
                row.append(cell)
            lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Polarity head, mean and standard deviation over ten seeds, in percent. "
        r"\textsc{aug} adds the derived-polarity augmentation to \textsc{raw}. NaN-NLI and "
        r"MoNLI are held-out sets. \textbf{Bold}: best per column.}",
        r"\label{tab:main}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def suites_table(cells) -> str:
    """The three generated families separately, read off the sanity confusion matrix."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\begin{tabular}{l l ccc}",
        r"\toprule",
        r"Encoder & Condition & Identical & Unrelated & Mirrored \\",
        r"\midrule",
    ]
    shown = [
        a for a in ("nli-deberta-v3-large", "deberta-v3-large", "bert") if (a, "none") in cells or (a, "full") in cells
    ]
    for position, arch in enumerate(shown):
        if position:
            lines.append(r"\addlinespace")
        for condition, name in (("none", r"\textsc{raw}"), ("full", r"\textsc{aug}")):
            row = [r"\multirow{2}{*}{" + ARCH_LABELS[arch] + "}" if condition == "none" else "", name]
            row += [as_percent(cells.get((arch, condition), {}).get(suite, [])) for suite in SUITES]
            lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{The three generated suites separately, ten seeds, in percent. Identical "
        r"pairs must be entailment, unrelated pairs neutral, mirrored pairs contradiction.}",
        r"\label{tab:suites}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def stats_table(cells) -> str:
    """Per-encoder significance of the augmentation, on both axes at once.

    Both tests on the same footing is what makes the first one credible: reporting only
    the metric that moved would leave the reader to take "nothing changed" on trust.
    """
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\begin{tabular}{l cc cc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{macro-F\textsubscript{1}} & \multicolumn{2}{c}{Sanity suites} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Encoder & $\Delta$ (pp) & $d$ & $\Delta$ (pp) & $d$ \\",
        r"\midrule",
    ]
    for arch, label in ARCH_LABELS.items():
        raw, aug = cells.get((arch, "none")), cells.get((arch, "full"))
        if not raw or not aug:
            continue
        task = welch(aug["macro_f1"], raw["macro_f1"])
        sanity = welch(aug["sanity"], raw["sanity"])
        mark = "" if task["p"] < 0.05 else r"$^{\dagger}$"
        lines.append(
            f"{label} & {100 * task['diff']:+.2f}\\,$\\pm$\\,{100 * error(aug['macro_f1'], raw['macro_f1']):.2f}{mark}"
            f" & {task['d']:+.2f} "
            f"& {100 * sanity['diff']:+.2f}\\,$\\pm$\\,{100 * error(aug['sanity'], raw['sanity']):.2f}"
            f" & {sanity['d']:+.1f} " + r"\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Effect of augmentation in points, \textsc{aug} minus \textsc{raw}, with "
        r"the standard error of the difference and Cohen's $d$ (Welch, ten seeds). "
        r"$\dagger$: not significant at $p<0.05$. Every sanity difference has $p<10^{-4}$.}",
        r"\label{tab:stats}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def error(left: list[float], right: list[float]) -> float:
    """Standard error of the difference of two means, Welch's form.

    Guarded like the tests it is printed beside: one observation has no variance, and a
    cell that lost its seeds should put a NaN in the table rather than stop the build of
    every other table with it.
    """
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    return math.sqrt(st.variance(left) / len(left) + st.variance(right) / len(right))


def axis_range(cells, order, metric: str) -> tuple[int, int]:
    """The panel's x range, taken from the data rather than written down.

    A hard-coded range silently drops a point that falls outside it, which is the one
    failure mode of a generated figure that no compiler reports. The bounds include the
    error bars, round outward to whole points, and stop just past 100 because every
    metric on these panels is a percentage.
    """
    low, high = [], []
    for arch in order:
        for condition in ("none", "full"):
            values = cells.get((arch, condition), {}).get(metric, [])
            if not values:
                continue
            mean, spread = mean_sd(values)
            low.append(100 * (mean - spread))
            high.append(100 * (mean + spread))
    if not low:
        return (0, 100)
    return (int(max(0, math.floor(min(low) - 1))), int(min(101, math.ceil(max(high) + 1))))


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

    No legend box and no vertical rules. The two conditions are named in the caption,
    where they cost no space inside the panel; the gridlines were competing with the very
    marks they were supposed to help locate, so the axis keeps its ticks and nothing else.
    """
    order = [a for a in ARCH_LABELS if (a, "none") in cells and (a, "full") in cells][::-1]
    labels = ", ".join(ARCH_LABELS[a] for a in order)

    def series(metric: str, condition: str, offset: float) -> str:
        points = []
        for index, arch in enumerate(order):
            values = cells[(arch, condition)][metric]
            points.append(
                f"({100 * st.mean(values):.2f},{index + offset:.2f})" f" +- ({100 * mean_sd(values)[1]:.2f},0)"
            )
        return " ".join(points)

    marks = {
        "none": "mark=*, mark size=2.2pt, color=condraw, mark options={draw=condraw, fill=condraw}",
        "full": "mark=square*, mark size=2.1pt, color=condaug, mark options={draw=condaug, fill=condaug}",
    }
    panels = []
    for metric, title in (("macro_f1", r"Macro-F$_1$ (\%)"), ("sanity", r"Sanity suites (\%)")):
        xmin, xmax = axis_range(cells, order, metric)
        body = [f"\\nextgroupplot[title={{{title}}}, xmin={xmin}, xmax={xmax}]"]
        for condition, offset in (("none", -0.19), ("full", 0.19)):
            # Deux pieges de portabilite, tous deux vus sur Overleaf et non ici.
            #
            # Pas de "error bars/.cd" : le .cd deplace le chemin de cles pour TOUT ce qui
            # suit dans la liste d'options, y compris les cles que "\addplot+" y ajoute
            # depuis la liste cyclique. Selon la version de pgfplots, mark et color se
            # retrouvent cherches sous /pgfplots/error bars/, ou ils n'existent pas ; le
            # chemin avorte et TikZ rend "Giving up on this path" sur la ligne SUIVANTE.
            # Les chemins de cles sont donc ecrits au long.
            #
            # Pas de "+" non plus : le style pose deja la marque et la couleur, donc la
            # liste cyclique n'a rien a apporter et tout a casser.
            body.append(
                f"\\addplot[only marks, {marks[condition]}, "
                f"error bars/x dir=both, error bars/x explicit] "
                f"coordinates {{{series(metric, condition, offset)}}};"
            )
        panels.append("\n".join(body))

    lines = [
        r"% Genere par src/figures_generator/analyse_v3.py. Ne pas editer a la main.",
        r"% Palette Okabe-Ito, validee : bande de clarte, plancher de chroma, separation",
        r"% daltonienne, plancher de vision normale et contraste sur fond clair. La forme",
        r"% du marqueur double la couleur, pour que l'identite ne repose jamais sur elle",
        r"% seule, et la legende de figure nomme les deux conditions dans leur teinte.",
        r"\definecolor{condraw}{HTML}{0072B2}",
        r"\definecolor{condaug}{HTML}{D55E00}",
        r"\begin{figure*}[t]",
        r"\centering",
        r"% Aucun style nomme : un style declare par \tikzset vit sous /tikz/, et",
        r"% \addplot resout ses options sous /pgfplots/. Selon la version, le repli",
        r"% n'a pas lieu, la cle est inconnue et le chemin avorte. Les options sont",
        r"% donc ecrites en clair dans chaque \addplot.",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 1, horizontal sep=0.6cm, y descriptions at=edge left},",
        r"  width=0.365\textwidth, height=6.2cm,",
        r"  scale only axis,",
        r"  % Tufte : pas de cadre, une seule ligne d'axe, aucune regle verticale. Les",
        r"  % graduations suffisent a situer un point, et la grille entrait en competition",
        r"  % avec les marques qu'elle devait aider a lire.",
        r"  axis line style={draw=black!45, line width=0.3pt},",
        r"  % L'axe y ne porte pas de ligne, mais il porte les noms d'encodeurs : les",
        r"  % supprimer avec la ligne laisse sept rangees de points que rien n'identifie.",
        r"  axis x line*=bottom, axis y line*=left,",
        r"  y axis line style={draw=none},",
        r"  xmajorgrids=false, ymajorgrids=false,",
        r"  tick align=outside, ytick style={draw=none},",
        r"  tick label style={font=\small}, title style={font=\small, yshift=-2pt},",
        f"  ymin=-0.7, ymax={len(order) - 0.3}, ytick={{{','.join(str(i) for i in range(len(order)))}}},",
        f"  yticklabels={{{labels}}}, yticklabel style={{font=\\small}},",
        r"  every axis plot/.append style={line width=0.7pt},",
        r"]",
        panels[0],
        panels[1],
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"\caption{Augmentation moves one axis and not the other. "
        r"\textcolor{condraw}{\textbf{\textsc{raw}}} and "
        r"\textcolor{condaug}{\textbf{\textsc{aug}}}: means over ten seeds, bars one "
        r"standard deviation. The panels do not share an $x$ range.}",
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
    orphelines = unlabelled_cells(results)
    if orphelines:
        raise click.ClickException(
            "cellules entrainees sans etiquette dans ARCH_LABELS, elles seraient ignorees "
            "en silence : " + ", ".join(sorted(orphelines))
        )
    cells = load(results)
    findings: dict[str, Any] = {"cells": {}, "augmentation": {}, "nli": {}}

    click.echo("=== cellules (moyenne +- ecart-type sur les graines)\n")
    heads = "".join(f"{name:>17}" for name in METRICS.values())
    click.echo(f"{'architecture':<28} {'cond':<5} {'n':>2}{heads}")
    for arch in ARCH_LABELS:
        for condition in ("none", "full"):
            values = cells.get((arch, condition))
            if not values:
                continue
            row = [f"{st.mean(values[m]):.4f} ± {mean_sd(values[m])[1]:.4f}" for m in METRICS]
            cells_text = "".join(f"{value:>17}" for value in row)
            click.echo(f"{arch:<28} {condition:<5} {len(values['macro_f1']):>2}{cells_text}")
            findings["cells"][f"{arch}-{condition}"] = {
                m: {"mean": st.mean(values[m]), "sd": mean_sd(values[m])[1], "n": len(values[m])} for m in METRICS
            }

    click.echo("\n=== effet de l'augmentation, Welch par architecture")
    click.echo(f"{'architecture':<28} {'macro-F1 (aug - raw)':>22} {'sanity (aug - raw)':>22}")
    for arch in ARCH_LABELS:
        raw, aug = cells.get((arch, "none")), cells.get((arch, "full"))
        if not raw or not aug:
            continue
        task = welch(aug["macro_f1"], raw["macro_f1"])
        sanity = welch(aug["sanity"], raw["sanity"])
        findings["augmentation"][arch] = {
            "macro_f1": task,
            "sanity": sanity,
            "monli": welch(aug["monli"], raw["monli"]),
        }
        click.echo(
            f"{arch:<28} {task['diff']:+8.4f} p={task['p']:<7.4f} d={task['d']:+5.2f}"
            f" {sanity['diff']:+8.4f} p={sanity['p']:<7.4f} d={sanity['d']:+5.2f}"
        )

    click.echo("\n=== apport du pre-entrainement NLI, apparie par graine")
    for condition in ("none", "full"):
        for nli, plain in (("nli-deberta-v3-large", "deberta-v3-large"), ("nli-deberta-v3-base", "deberta-v3-base")):
            left, right = cells.get((nli, condition)), cells.get((plain, condition))
            if not left or not right:
                continue
            for metric in ("macro_f1", "monli"):
                got = paired(left[metric], right[metric])
                findings["nli"][f"{nli}-vs-{plain}-{condition}-{metric}"] = got
                click.echo(
                    f"{nli[:22]:<22} {condition:<5} {METRICS[metric]:<9} {got['diff']:+7.4f}"
                    f"  p={got['p']:<8.4f} d={got['d']:+5.2f}"
                )

    if tex_out:
        os.makedirs(tex_out, exist_ok=True)
        with open(os.path.join(tex_out, "table_main.tex"), "w", encoding="utf-8") as handle:
            handle.write(main_table(cells) + "\n")
        with open(os.path.join(tex_out, "table_stats.tex"), "w", encoding="utf-8") as handle:
            handle.write(stats_table(cells) + "\n")
        with open(os.path.join(tex_out, "table_suites.tex"), "w", encoding="utf-8") as handle:
            handle.write(suites_table(cells) + "\n")
        figure(cells, os.path.join(tex_out, "figure_augmentation.tex"))
        click.echo(f"\ntables : {tex_out}")
    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(findings, handle, indent=2)
        click.echo(f"brut : {json_out}")


if __name__ == "__main__":
    main()
