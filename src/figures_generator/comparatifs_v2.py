"""Comparison figures and LaTeX tables for the v2 grid, from local JSON or from wandb.

One data shape, two sources, three outputs. The point of the single shape is that the
figure, the table and the wandb view can never disagree: they are three renderings of the
same list of ``RunResult``.

Sources
    ``--source local``  reads ``results/`` the way ``analyze_v2_experiment.py`` does.
    ``--source wandb``  reads the same runs back from the wandb projects they logged to.

Both are kept because they fail differently. The JSON is written at the very end of a run,
so a crashed run leaves nothing; wandb holds the partial history of a run that died, but
needs the network and the run to have reached its final summary. When a table looks wrong,
building it from the other source is the fastest way to tell a reporting bug from a
training bug.

Run::

    PYTHONPATH=src python src/figures_generator/comparatifs_v2.py --source local
    PYTHONPATH=src python src/figures_generator/comparatifs_v2.py --source wandb --wandb-upload
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Optional

import click
import matplotlib

matplotlib.use("Agg")  # No display on the training hosts, and none needed to write a PDF.
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

try:  # PYTHONPATH=src.
    from figures_generator.analyze_v2_experiment import CONDITION_LABELS, MODE_LABELS, RunResult, load_runs
except ImportError:  # pragma: no cover - script run from inside ``src/figures_generator``.
    from analyze_v2_experiment import CONDITION_LABELS, MODE_LABELS, RunResult, load_runs  # type: ignore

#: Architectures in the order they mean something: the v1 baseline first, the model the
#: paper proposes last. Alphabetical order would put ``bert`` next to ``deberta`` for no
#: reason at all.
ARCH_ORDER = [
    "bert-base-uncased",
    "stsb-roberta-base",
    "DeBERTa-v3-base-mnli-fever-anli",
    "deberta-v3-base",
    "deberta-v3-large",
]
#: Grid variants, and the two diagnostics that only exist on the reference architecture.
VARIANT_ORDER = ["a_none", "a_full", "b_none", "b_full", "c_none", "c_full", "d_none", "d_full"]
GRID_VARIANTS = ["c_none", "c_full", "d_none", "d_full"]
#: "linear" is the published v1 model, whose head IS its logit. It shows up as soon as
#: an evaluation of the published weights lands in results/, and a figure that colours
#: by head must know about it or it raises on a missing palette key.
HEAD_ORDER = ["linear", "sigmoid", "clamped"]

#: Targets of PRODUIT.md, drawn on every figure that carries the metric they bound.
TARGET_PEARSON = 0.914
TARGET_RMSE = 15.0


def set_theme() -> None:
    """One theme for every figure of the paper.

    Set once, here, rather than per figure: a figure that styles itself is a figure that
    will drift from its neighbours the day someone edits only one of them.
    """
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.05,
        rc={
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.edgecolor": "#33333a",
            "axes.labelcolor": "#1c1c22",
            "axes.titleweight": "semibold",
            "grid.color": "#d8d8e0",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        },
    )


#: Colour-blind safe, and stable across figures so a reader learns the mapping once.
PALETTE_HEAD = {"linear": "#8a8f98", "sigmoid": "#b0879b", "clamped": "#2f6f8f"}
PALETTE_MODE = {"none": "#7c9cb0", "full": "#c08552"}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def runs_to_frame(runs: Iterable[RunResult]) -> pd.DataFrame:
    """Turn the run objects into the one table every renderer reads."""
    rows = []
    for run in runs:
        rows.append(
            {
                "arch": run.arch,
                "head": run.head,
                "seed": run.seed,
                "variant": run.variant,
                "condition": run.condition,
                "mode": run.mode,
                "condition_label": CONDITION_LABELS.get(run.condition, run.condition),
                "mode_label": MODE_LABELS.get(run.mode, run.mode),
                "pearson": run.pearson,
                "rmse": run.rmse,
                "rmse_floor": run.rmse_floor,
                "r2": run.r2,
                "pred_mean": run.pred_mean,
                "pred_std": run.pred_std,
                "identical_mean": run.identical_mean,
                "identical_ratio_95": run.identical_ratio_95,
                "unrelated_mean": run.unrelated_mean,
                "unrelated_ratio_5": run.unrelated_ratio_5,
                "objective": run.objective,
                "epochs": run.epochs,
                "train_rows": run.train_rows,
                "diverged": run.diverged,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["arch"] = pd.Categorical(frame["arch"], categories=_ordered(frame["arch"], ARCH_ORDER), ordered=True)
    frame["variant"] = pd.Categorical(
        frame["variant"], categories=_ordered(frame["variant"], VARIANT_ORDER), ordered=True
    )
    frame["head"] = pd.Categorical(frame["head"], categories=_ordered(frame["head"], HEAD_ORDER), ordered=True)
    return frame.sort_values(["arch", "head", "variant"]).reset_index(drop=True)


def _ordered(values: pd.Series, preferred: list[str]) -> list[str]:
    """Preferred order first, then whatever else showed up, so a new architecture never
    silently disappears from a figure because it is not in the hard-coded list."""
    seen = list(dict.fromkeys(values.dropna().astype(str)))
    return [v for v in preferred if v in seen] + [v for v in seen if v not in preferred]



#: Metrics aggregated across seeds. Everything else in a run is either constant across
#: seeds (the corpus, the architecture) or not worth a standard deviation (the row counts).
AGGREGATED = ["objective", "pearson", "rmse", "r2", "identical_mean", "identical_ratio_95",
              "unrelated_mean", "unrelated_ratio_5", "pred_mean", "pred_std", "epochs"]


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation and count per cell, over the seeds of that cell.

    The article reports mean and standard deviation over seeds 42 to 51, and it has to:
    the gaps measured on a single seed run from 0.007 to 0.014 in Pearson, which is the
    order of an initialisation draw. A number without its spread cannot rank anything.

    ``n`` is carried into every table because a mean over two seeds and a mean over ten
    are not the same claim, and a half-finished sweep must say which one it is showing.
    """
    if frame.empty:
        return frame
    grouped = frame.groupby(["arch", "head", "variant"], observed=True)
    out = grouped[AGGREGATED].agg(["mean", "std"])
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    out["n"] = grouped.size()
    return out.reset_index().dropna(subset=["pearson_mean"])


def pm(mean: float, std: float, digits: int = 3, latex: bool = False) -> str:
    """One cell as ``mean +/- std``, or just the mean when a single seed ran.

    A standard deviation over one run is not zero, it is undefined; printing ``0.000``
    would claim a reproducibility that was never measured.
    """
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{digits}f}"
    sep = r" $\pm$ " if latex else " ± "
    return f"{mean:.{digits}f}{sep}{std:.{digits}f}"


def load_from_wandb(entity: str, prefix: str) -> list[RunResult]:
    """Rebuild the run list from the wandb projects the grid logged to.

    Every grid run logs to ``meaningbert-v2-<arch>-<head>`` and phase 1 to
    ``meaningbert-v2``, so the projects to read are found by prefix rather than listed by
    hand: an architecture added to the grid must not require editing this file.
    """
    import wandb  # Imported here so the local path never needs the network.

    api = wandb.Api()
    runs: list[RunResult] = []
    for project in api.projects(entity):
        if not project.name.startswith(prefix):
            continue
        for run in api.runs(f"{entity}/{project.name}"):
            built = _run_from_wandb(run)
            if built is not None:
                runs.append(built)
    return runs


def _run_from_wandb(run) -> Optional[RunResult]:  # noqa: ANN001 - wandb's Run has no public type
    """One wandb run into a RunResult, or None when it never reached its test evaluation."""
    summary = {k: v for k, v in run.summary.items()} if run.summary is not None else {}
    config = run.config or {}
    if "test_pearson_corr" not in summary:
        return None  # Crashed, or still running: no test evaluation, nothing to compare.

    # The variant is not a config field; it travels in the run name, which the trainer
    # builds as ``..._<variant>_head<head>``. Parsing it is ugly but it is the only place
    # the information exists on runs already finished, and rerunning them to add a field
    # would cost more than this function is worth.
    variant = "unknown"
    for candidate in VARIANT_ORDER:
        if f"_{candidate}_" in run.name or run.name.endswith(f"_{candidate}"):
            variant = candidate
            break
    condition, _, mode = variant.partition("_")

    def number(key: str) -> float:
        try:
            return float(summary[key])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    return RunResult(
        arch=str(config.get("checkpoint", "")).split("/")[-1] or "unknown",
        variant=variant,
        condition=condition,
        mode=mode,
        pearson=number("test_pearson_corr"),
        rmse=number("test_rmse"),
        r2=number("test_R2"),
        pred_mean=number("test_mean_score"),
        pred_std=number("test_st_dev_score"),
        identical_ratio_95=number("test/identical_sentences_ratio_95"),
        unrelated_ratio_5=number("test/unrelated_sentences_ratio_5"),
        epochs=number("epoch"),
        train_rows=0,
        diverged=number("test_diverged") == 1.0,
        head=str(config.get("output_head") or "unknown"),
        seed=int(config.get("seed") or -1),
        identical_mean=number("test/identical_sentences_mean_score"),
        unrelated_mean=number("test/unrelated_sentences_mean_score"),
    )


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def save(fig, path: str) -> str:
    """Write the figure three times, once per consumer.

    PDF for the paper, SVG for the HTML report, PNG for wandb. SVG and not PNG in the
    report because these figures print small numbers inside the cells and a raster at any
    sane file size makes them mush when the reader zooms; PNG all the same because
    ``wandb.Image`` only accepts a raster.
    """
    stem = os.path.splitext(path)[0]
    fig.savefig(path)
    fig.savefig(stem + ".svg")
    fig.savefig(stem + ".png")
    plt.close(fig)
    return path


def figure_head_effect(frame: pd.DataFrame, path: str) -> Optional[str]:
    """Sigmoid against clamped, on the runs where both heads exist.

    A dumbbell and not a pair of bars: the quantity of interest is the MOVE between the
    two heads, and a bar chart makes the reader compute it by eye across a gap.
    """
    both = frame[frame["variant"].isin(GRID_VARIANTS)]
    pivot = both.pivot_table(index=["arch", "variant"], columns="head", values=["pearson", "identical_ratio_95"],
                             observed=True)
    pivot = pivot.dropna(subset=[("pearson", "sigmoid"), ("pearson", "clamped")], how="any")
    if pivot.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.55 * len(pivot) + 2.4), sharey=True)
    labels = [f"{arch}\n{variant}" for arch, variant in pivot.index]
    y = range(len(pivot))

    for ax, metric, title, xlabel in (
        (axes[0], "pearson", "Correlation", "Pearson $r$ on the test set"),
        (axes[1], "identical_ratio_95", "Sanity check", "Identical pairs scored above 95 (%)"),
    ):
        left = pivot[(metric, "sigmoid")].to_numpy()
        right = pivot[(metric, "clamped")].to_numpy()
        ax.hlines(list(y), left, right, color="#9aa0aa", linewidth=1.6, zorder=1)
        ax.scatter(left, list(y), s=70, color=PALETTE_HEAD["sigmoid"], label="sigmoid", zorder=2)
        ax.scatter(right, list(y), s=70, color=PALETTE_HEAD["clamped"], label="clamped", zorder=2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0].axvline(TARGET_PEARSON, color="#a03030", linestyle="--", linewidth=1.1)
    axes[0].text(TARGET_PEARSON, -0.75, " target", color="#a03030", fontsize=8, va="top")
    axes[0].set_yticks(list(y), labels)
    axes[1].legend(loc="lower left", title="output head")
    fig.suptitle("The clamped head buys the sanity check and costs no correlation", y=1.01)
    fig.tight_layout()
    return save(fig, path)


def figure_objective_plane(frame: pd.DataFrame, path: str) -> Optional[str]:
    """Correlation against the identical-pair check, the plane the objective lives in.

    The objective is a product of three terms, so a single ranked bar hides which term is
    the weak one. Here a run's weakness is its position: far right and low means a model
    that correlates and fails the check.
    """
    usable = frame.dropna(subset=["pearson", "identical_ratio_95"])
    if usable.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    sns.scatterplot(
        data=usable,
        x="pearson",
        y="identical_ratio_95",
        hue="head",
        style="arch",
        s=130,
        palette=PALETTE_HEAD,
        ax=ax,
    )
    ax.axhline(95, color="#4a7a4a", linestyle=":", linewidth=1.2)
    ax.text(usable["pearson"].min(), 95.6, "95 % of identical pairs above 95", color="#4a7a4a", fontsize=8)
    ax.axvline(TARGET_PEARSON, color="#a03030", linestyle="--", linewidth=1.1)
    ax.text(TARGET_PEARSON - 0.002, 4, "PRODUIT.md target", color="#a03030", fontsize=8, rotation=90, ha="right")
    ax.set_xlabel("Pearson $r$ on the test set")
    ax.set_ylabel("Identical pairs scored above 95 (%)")
    ax.set_title("Where each run sits in the objective plane")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return save(fig, path)


def figure_grid_heatmap(frame: pd.DataFrame, path: str, head: str = "clamped") -> Optional[str]:
    """Architecture against corpus variant, one cell per run, blank where nothing ran yet.

    A heatmap and not a grouped bar chart because the grid is half empty while it runs,
    and a missing bar reads as a zero while a missing cell reads as missing.
    """
    subset = frame[(frame["head"] == head) & frame["variant"].isin(GRID_VARIANTS)]
    if subset.empty:
        return None
    pivot = subset.pivot_table(index="arch", columns="variant", values="objective", observed=True)
    annot = subset.pivot_table(index="arch", columns="variant", values="pearson", observed=True)
    labels = annot.map(lambda v: "" if not isinstance(v, float) or math.isnan(v) else f"{v:.3f}")

    fig, ax = plt.subplots(figsize=(1.55 * max(1, pivot.shape[1]) + 3.6, 0.85 * max(1, pivot.shape[0]) + 2.2))
    sns.heatmap(
        pivot,
        annot=labels,
        fmt="",
        cmap="crest",
        vmin=0,
        vmax=1,
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "objective = $r$ x (identical>95) x (unrelated<5)"},
        ax=ax,
    )
    ax.set_title(f"Grid, {head} head: colour is the objective, the number is Pearson $r$")
    ax.set_xlabel("corpus variant")
    ax.set_ylabel("")
    fig.tight_layout()
    return save(fig, path)


def figure_corpus_and_augmentation(frame: pd.DataFrame, path: str, head: str = "clamped") -> Optional[str]:
    """The two factors the experiment was built to measure, side by side.

    RMSE gets its floor drawn on it. Without that line the reader cannot tell an RMSE that
    improved from an RMSE that merely followed the correlation.
    """
    subset = frame[(frame["head"] == head) & frame["variant"].isin(GRID_VARIANTS)].copy()
    if subset.empty:
        return None
    subset["corpus"] = subset["condition"].map({"c": "v1 corrected (c)", "d": "v2 corpora (d)"})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    sns.barplot(data=subset, x="corpus", y="pearson", hue="mode", palette=PALETTE_MODE, ax=axes[0], errorbar="sd", capsize=0.12)
    axes[0].axhline(TARGET_PEARSON, color="#a03030", linestyle="--", linewidth=1.1)
    axes[0].set_ylim(0.7, max(0.95, float(subset["pearson"].max()) + 0.03))
    axes[0].set_ylabel("Pearson $r$")
    axes[0].set_title("Correlation")

    sns.barplot(data=subset, x="corpus", y="rmse", hue="mode", palette=PALETTE_MODE, ax=axes[1], errorbar="sd", capsize=0.12)
    sns.pointplot(
        data=subset, x="corpus", y="rmse_floor", color="#33333a", linestyles="", markers="_",
        markersize=28, ax=axes[1],
    )
    axes[1].axhline(TARGET_RMSE, color="#a03030", linestyle="--", linewidth=1.1)
    axes[1].set_ylabel("RMSE (dash: floor at this correlation, red: target)")
    axes[1].set_title("Error")

    for ax in axes:
        ax.set_xlabel("")
        ax.legend(title="augmentation", loc="lower left", fontsize=8)
    fig.suptitle(f"Corpus and augmentation, {head} head, averaged over the architectures that finished", y=1.02)
    fig.tight_layout()
    return save(fig, path)


# --------------------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------------------
def _tex(value: str) -> str:
    return value.replace("_", r"\_")


def _cell(value: float, digits: int = 3, bold: bool = False) -> str:
    if value is None or not isinstance(value, float) or math.isnan(value):
        return "--"
    text = f"{value:.{digits}f}"
    return r"\textbf{" + text + "}" if bold else text


def table_head_effect(frame: pd.DataFrame, path: str) -> Optional[str]:
    """LaTeX table of the one factor measured under both heads."""
    subset = frame[frame["variant"].isin(GRID_VARIANTS)]
    pivot = subset.pivot_table(
        index=["arch", "variant"],
        columns="head",
        values=["pearson", "rmse", "identical_mean", "identical_ratio_95"],
        observed=True,
    ).dropna(subset=[("pearson", "sigmoid"), ("pearson", "clamped")], how="any")
    if pivot.empty:
        return None

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Output head, all else held equal. The sigmoid maps to the open interval "
        r"$(0, 100)$, so it cannot emit the two values about half the training labels take. "
        r"The clamped head reaches them. Correlation is unchanged; the identical-pair check is not.}",
        r"\label{tab:v2-output-head}",
        r"\begin{tabular}{l l c c c c c c}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Pearson $r$} & \multicolumn{2}{c}{Identical mean} "
        r"& \multicolumn{2}{c}{Identical $>95$ (\%)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Architecture & Variant & sigmoid & clamped & sigmoid & clamped & sigmoid & clamped \\",
        r"\midrule",
    ]
    for (arch, variant), row in pivot.iterrows():
        lines.append(
            " & ".join(
                [
                    _tex(str(arch)),
                    _tex(str(variant)),
                    _cell(row[("pearson", "sigmoid")]),
                    _cell(row[("pearson", "clamped")], bold=row[("pearson", "clamped")] >= row[("pearson", "sigmoid")]),
                    _cell(row[("identical_mean", "sigmoid")], 2),
                    _cell(row[("identical_mean", "clamped")], 2, bold=True),
                    _cell(row[("identical_ratio_95", "sigmoid")], 1),
                    _cell(row[("identical_ratio_95", "clamped")], 1, bold=True),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def table_grid(frame: pd.DataFrame, path: str, head: str = "clamped") -> Optional[str]:
    """LaTeX table of the grid itself, best value of each column in bold."""
    subset = frame[(frame["head"] == head) & frame["variant"].isin(GRID_VARIANTS)]
    if subset.empty:
        return None
    best = {
        "objective": subset["objective"].max(),
        "pearson": subset["pearson"].max(),
        "rmse": subset["rmse"].min(),
        "identical_ratio_95": subset["identical_ratio_95"].max(),
        "unrelated_ratio_5": subset["unrelated_ratio_5"].max(),
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{The v2 grid with the " + head + r" head. The objective is the product "
        r"$r \times (\text{identical}>95) \times (\text{unrelated}<5)$: a product and not an "
        r"average, so that one weak term drags the whole score down instead of being hidden. "
        r"The RMSE floor is what the best affine rescaling of these predictions would reach "
        r"at this correlation.}",
        r"\label{tab:v2-grid-" + head + "}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l l c c c c c c c}",
        r"\toprule",
        r"Architecture & Variant & Objective & Pearson $r$ & RMSE & RMSE floor & $R^2$ "
        r"& Identical $>95$ (\%) & Unrelated $<5$ (\%) \\",
        r"\midrule",
    ]
    previous = None
    for _, row in subset.sort_values(["arch", "variant"]).iterrows():
        if previous is not None and row["arch"] != previous:
            lines.append(r"\midrule")
        previous = row["arch"]
        lines.append(
            " & ".join(
                [
                    _tex(str(row["arch"])),
                    _tex(str(row["variant"])),
                    _cell(row["objective"], bold=row["objective"] == best["objective"]),
                    _cell(row["pearson"], bold=row["pearson"] == best["pearson"]),
                    _cell(row["rmse"], 2, bold=row["rmse"] == best["rmse"]),
                    _cell(row["rmse_floor"], 2),
                    _cell(row["r2"]),
                    _cell(row["identical_ratio_95"], 1, bold=row["identical_ratio_95"] == best["identical_ratio_95"]),
                    _cell(row["unrelated_ratio_5"], 1, bold=row["unrelated_ratio_5"] == best["unrelated_ratio_5"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def table_seeds(frame: pd.DataFrame, path: str, head: str = "clamped") -> Optional[str]:
    """The table the article publishes: mean and standard deviation over seeds.

    Returns ``None`` while every cell still holds a single seed. A table of means over one
    run each would look like a multi-seed result and is not one; better no table than a
    table that overstates what was measured.
    """
    agg = aggregate(frame[(frame["head"] == head) & frame["variant"].isin(GRID_VARIANTS)])
    if agg.empty or int(agg["n"].max()) < 2:
        return None

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{The v2 grid with the " + head + r" head, mean $\pm$ standard deviation over "
        r"seeds 42 to 51, the protocol of the original article. $n$ is the number of seeds that "
        r"finished for that cell; a cell with $n = 1$ carries no standard deviation because a "
        r"spread over one run is undefined, not zero.}",
        r"\label{tab:v2-seeds-" + head + "}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l l c c c c c}",
        r"\toprule",
        r"Architecture & Variant & $n$ & Objective & Pearson $r$ & RMSE & Identical $>95$ (\%) \\",
        r"\midrule",
    ]
    previous = None
    for _, row in agg.sort_values(["arch", "variant"]).iterrows():
        if previous is not None and row["arch"] != previous:
            lines.append(r"\midrule")
        previous = row["arch"]
        lines.append(
            " & ".join(
                [
                    _tex(str(row["arch"])), _tex(str(row["variant"])), str(int(row["n"])),
                    pm(row["objective_mean"], row["objective_std"], 3, latex=True),
                    pm(row["pearson_mean"], row["pearson_std"], 3, latex=True),
                    pm(row["rmse_mean"], row["rmse_std"], 2, latex=True),
                    pm(row["identical_ratio_95_mean"], row["identical_ratio_95_std"], 1, latex=True),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


# --------------------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------------------
HTML_CSS = """
:root {
  --ink: #16161a; --muted: #5d6270; --rule: #d9dae1; --bg: #ffffff; --panel: #f7f8fa;
  --accent: #2f6f8f; --warn: #a03030; --ok: #3f7a4f;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink: #e9e9ee; --muted: #a0a4b0; --rule: #33343c; --bg: #17181c; --panel: #1f2026;
    --accent: #7fb7d4; --warn: #e08a8a; --ok: #8fc79f;
  }
}
:root[data-theme="dark"] {
  --ink: #e9e9ee; --muted: #a0a4b0; --rule: #33343c; --bg: #17181c; --panel: #1f2026;
  --accent: #7fb7d4; --warn: #e08a8a; --ok: #8fc79f;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.62 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
main { max-width: 1120px; margin: 0 auto; padding: 48px 16px 96px; }
h1 { font-size: 1.9rem; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 1.25rem; margin: 56px 0 4px; letter-spacing: -0.005em; }
h2 .num { color: var(--muted); font-variant-numeric: tabular-nums; margin-right: 10px; font-weight: 500; }
.sub { color: var(--muted); margin: 0 0 22px; }
.key { border-left: 3px solid var(--accent); background: var(--panel); padding: 12px 16px; margin: 16px 0 24px; }
.key b { font-weight: 600; }
figure { margin: 0 0 8px; background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 14px; overflow-x: auto; }
figure svg { max-width: 100%; height: auto; display: block; margin: 0 auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 0.92rem; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
thead th { border-bottom: 2px solid var(--ink); font-weight: 600; }
tbody tr:hover { background: var(--panel); }
td.best { font-weight: 700; color: var(--accent); }
td.fail { color: var(--warn); }
.wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px; padding: 4px 10px; }
.meta { color: var(--muted); font-size: 0.86rem; margin-top: 10px; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 999px; border: 1px solid var(--rule); font-size: 0.8rem; color: var(--muted); margin-right: 6px; }
@media (max-width: 720px) { main { padding: 28px 16px 64px; } h1 { font-size: 1.5rem; } }
"""


def _svg(path: str) -> str:
    """Inline the SVG so the report is a single file that survives being emailed."""
    candidate = os.path.splitext(path)[0] + ".svg"
    if not os.path.exists(candidate):
        return f"<p class='meta'>figure absente : {os.path.basename(candidate)}</p>"
    with open(candidate, encoding="utf-8") as handle:
        markup = handle.read()
    return markup[markup.index("<svg") :]


def _html_table(frame: pd.DataFrame, columns: list[tuple[str, str, int]], best: dict[str, str]) -> str:
    """Render a frame as an HTML table, bolding the best cell of the columns that have one.

    *best* maps a column to ``max`` or ``min``; a column absent from it has no winner, which
    is the honest rendering for something like the predicted mean.
    """
    winners = {}
    for column, direction in best.items():
        if column in frame and frame[column].notna().any():
            winners[column] = frame[column].max() if direction == "max" else frame[column].min()

    head = "".join(f"<th>{title}</th>" for title, _, _ in columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for _, key, digits in columns:
            value = row.get(key)
            if isinstance(value, float) and math.isnan(value):
                cells.append("<td>--</td>")
                continue
            if isinstance(value, float):
                css = " class='best'" if key in winners and value == winners[key] else ""
                cells.append(f"<td{css}>{value:.{digits}f}</td>")
            else:
                cells.append(f"<td>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def html_report(frame: pd.DataFrame, figures_dir: str, path: str, done: int, total: int) -> str:
    """One self-contained page holding every figure and every table, in reading order."""
    grid = frame[(frame["head"] == "clamped") & frame["variant"].isin(GRID_VARIANTS)]
    both = frame[frame["variant"].isin(GRID_VARIANTS)]
    pivot = both.pivot_table(
        index=["arch", "variant"], columns="head",
        values=["pearson", "identical_mean", "identical_ratio_95"], observed=True,
    ).dropna(subset=[("pearson", "sigmoid"), ("pearson", "clamped")], how="any")

    head_rows = []
    for (arch, variant), row in pivot.iterrows():
        head_rows.append(
            {
                "arch": str(arch), "variant": str(variant),
                "p_sig": row[("pearson", "sigmoid")], "p_cla": row[("pearson", "clamped")],
                "im_sig": row[("identical_mean", "sigmoid")], "im_cla": row[("identical_mean", "clamped")],
                "i95_sig": row[("identical_ratio_95", "sigmoid")], "i95_cla": row[("identical_ratio_95", "clamped")],
            }
        )
    head_frame = pd.DataFrame(head_rows)

    best_line = "aucun run termine"
    if not grid.empty:
        best = grid.loc[grid["objective"].idxmax()]
        best_line = (
            f"{best['arch']} / {best['variant']} : objectif {best['objective']:.3f}, "
            f"Pearson {best['pearson']:.3f}, RMSE {best['rmse']:.2f}, "
            f"identiques &gt;95 {best['identical_ratio_95']:.1f} %, non reliees &lt;5 {best['unrelated_ratio_5']:.1f} %"
        )

    sections = [
        (
            "1", "Tete de sortie",
            "Le seul facteur mesure sous les deux tetes, tout le reste egal : meme architecture, "
            "meme corpus, meme graine.",
            "<b>Comment lire.</b> La sigmoide a pour image l'intervalle OUVERT (0, 100) : elle ne peut "
            "jamais emettre 0 ni 100, alors qu'environ la moitie des etiquettes d'entrainement valent "
            "exactement l'un des deux. La tete clamped atteint les bornes. Si le changement etait "
            "gratuit, la correlation ne bougerait pas et le test des paires identiques monterait.",
            _svg(os.path.join(figures_dir, "v2-tete-de-sortie.pdf")),
            _html_table(
                head_frame,
                [
                    ("Architecture", "arch", 0), ("Variante", "variant", 0),
                    ("Pearson sigmoide", "p_sig", 3), ("Pearson clamped", "p_cla", 3),
                    ("Identiques moy. sigmoide", "im_sig", 2), ("Identiques moy. clamped", "im_cla", 2),
                    ("Identiques &gt;95 sigmoide (%)", "i95_sig", 1), ("Identiques &gt;95 clamped (%)", "i95_cla", 1),
                ],
                {},
            ) if not head_frame.empty else "<p class='meta'>pas encore de paire complete</p>",
        ),
        (
            "2", "Plan de l'objectif",
            "Chaque run place selon ses deux dimensions les plus contraignantes.",
            "<b>Comment lire.</b> L'objectif est un PRODUIT de trois termes, donc un run peut etre a "
            "droite, tres correle, et rester inutilisable parce qu'il est en bas. La ligne verticale "
            "rouge est la cible de correlation de PRODUIT.md, la ligne horizontale le seuil du test "
            "de bon sens. La zone utile est le coin superieur droit, et personne n'y est encore.",
            _svg(os.path.join(figures_dir, "v2-plan-objectif.pdf")),
            "",
        ),
        (
            "3", "La grille",
            "Architecture par variante de corpus, tete clamped. Les cases vides sont les runs qui "
            "n'ont pas encore tourne.",
            "<b>Comment lire.</b> La couleur est l'objectif, le nombre imprime est le Pearson. Une "
            "case pale avec un nombre eleve est exactement le piege que l'objectif sert a reveler : "
            "une bonne correlation annulee par un test de bon sens rate.",
            _svg(os.path.join(figures_dir, "v2-grille.pdf")),
            _html_table(
                grid,
                [
                    ("Architecture", "arch", 0), ("Variante", "variant", 0),
                    ("Objectif", "objective", 3), ("Pearson", "pearson", 3),
                    ("RMSE", "rmse", 2), ("Plancher RMSE", "rmse_floor", 2), ("R2", "r2", 3),
                    ("Identiques &gt;95 (%)", "identical_ratio_95", 1),
                    ("Non reliees &lt;5 (%)", "unrelated_ratio_5", 1),
                    ("Moy. predite", "pred_mean", 1), ("Ecart-type predit", "pred_std", 1),
                ],
                {"objective": "max", "pearson": "max", "rmse": "min",
                 "identical_ratio_95": "max", "unrelated_ratio_5": "max"},
            ) if not grid.empty else "<p class='meta'>grille vide</p>",
        ),
        (
            "4", "Corpus et augmentation",
            "Les deux facteurs que l'experience a ete construite pour mesurer.",
            "<b>Comment lire.</b> Le plancher de RMSE est ce que la meilleure remise a l'echelle "
            "affine de ces predictions atteindrait a cette correlation. Une RMSE collee a son "
            "plancher ne s'ameliorera pas par calibration : il faut une meilleure correlation.",
            _svg(os.path.join(figures_dir, "v2-corpus-augmentation.pdf")),
            "",
        ),
    ]

    parts = [
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Grille v2 MeaningBERT</title>",
        f"<style>{HTML_CSS}</style></head><body><main>",
        "<h1>Grille v2 MeaningBERT</h1>",
        f"<p class='sub'>Comparatif des runs termines. <span class='badge'>{done} runs de grille sur {total}</span>"
        f"<span class='badge'>cible : Pearson &ge; {TARGET_PEARSON}, RMSE &lt; {TARGET_RMSE:.0f}</span></p>",
        f"<div class='key'><b>Meilleure configuration a ce jour.</b> {best_line}</div>",
    ]
    for number, title, sub, key, svg, table in sections:
        parts.append(f"<h2><span class='num'>{number}</span>{title}</h2>")
        parts.append(f"<p class='sub'>{sub}</p>")
        parts.append(f"<div class='key'>{key}</div>")
        parts.append(f"<figure>{svg}</figure>")
        if table:
            parts.append(table)
    agg = aggregate(grid)
    if not agg.empty and int(agg["n"].max()) >= 2:
        rows = []
        for _, row in agg.sort_values(["arch", "variant"]).iterrows():
            rows.append(
                "<tr><td>" + "</td><td>".join([
                    str(row["arch"]), str(row["variant"]), str(int(row["n"])),
                    pm(row["objective_mean"], row["objective_std"], 3),
                    pm(row["pearson_mean"], row["pearson_std"], 3),
                    pm(row["rmse_mean"], row["rmse_std"], 2),
                    pm(row["identical_mean_mean"], row["identical_mean_std"], 2),
                    pm(row["identical_ratio_95_mean"], row["identical_ratio_95_std"], 1),
                ]) + "</td></tr>"
            )
        parts.append("<h2><span class='num'>5</span>Moyennes sur les graines</h2>")
        parts.append(
            "<p class='sub'>Protocole de l'article original : graines 42 a 51.</p>"
            "<div class='key'><b>Comment lire.</b> n est le nombre de graines terminees pour "
            "cette cellule. Une cellule a n = 1 ne porte pas d'ecart-type : une dispersion sur "
            "un seul tirage n'est pas nulle, elle est indefinie. Tant que deux cellules se "
            "chevauchent a un ecart-type, l'article ne peut pas les departager.</div>"
        )
        parts.append(
            "<div class='wrap'><table><thead><tr>"
            "<th>Architecture</th><th>Variante</th><th>n</th><th>Objectif</th><th>Pearson</th>"
            "<th>RMSE</th><th>Identiques moy.</th><th>Identiques &gt;95 (%)</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        )
    parts.append(
        "<p class='meta'>Les conditions a et b n'apparaissent pas dans la grille : ce sont les "
        "diagnostics de la fuite par phrase source (H5) et des etiquettes permutees (H6), et leur "
        "correlation est gonflee par la fuite. Elles vivent dans results/RESULTATS.md.</p>"
    )
    parts.append("</main></body></html>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    return path


# --------------------------------------------------------------------------------------
# wandb
# --------------------------------------------------------------------------------------
def upload_to_wandb(frame: pd.DataFrame, figures: list[str], entity: str, project: str) -> str:
    """Publish the comparison itself as a wandb run, next to the runs it compares.

    The grid spreads over one project per architecture and per head, which is right for
    training and useless for reading: no wandb view spans projects. A single run holding
    the joined table and the figures is the cross-project view that does not exist
    otherwise.
    """
    import wandb

    from datetime import datetime

    run = wandb.init(
        entity=entity,
        project=project,
        job_type="comparison",
        name=f"grille-v2-{datetime.now():%Y%m%d-%H%M}",
        reinit=True,
    )
    table = wandb.Table(dataframe=frame.astype({"arch": str, "variant": str, "head": str}))
    payload: dict = {"grid/all_runs": table}
    for path in figures:
        stem = os.path.splitext(path)[0]
        payload[f"figure/{os.path.basename(stem)}"] = wandb.Image(stem + ".png")
    finished = frame[frame["head"] == "clamped"]
    payload["grid/finished_runs"] = int(len(finished))
    if not finished.empty:
        best = finished.loc[finished["objective"].idxmax()]
        payload["best/arch"] = str(best["arch"])
        payload["best/variant"] = str(best["variant"])
        payload["best/objective"] = float(best["objective"])
        payload["best/pearson"] = float(best["pearson"])
    run.log(payload)
    url = run.url
    run.finish()
    return url


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
@click.command()
@click.option("--source", type=click.Choice(["local", "wandb"]), default="local", help="Where to read the runs from.")
@click.option("--runs-dir", default="results", help="Root of the run JSONs, for --source local.")
@click.option("--entity", default="davebulaval", help="wandb entity.")
@click.option("--project-prefix", default="meaningbert-v2", help="Projects to read, for --source wandb.")
@click.option("--figures-dir", default="results/figures", help="Where the figures go.")
@click.option("--tables-dir", default="results/tables", help="Where the LaTeX tables go.")
@click.option("--html-out", default="results/rapport-v2.html", help="Self-contained HTML report.")
@click.option("--wandb-upload/--no-wandb-upload", default=False, help="Publish the comparison to wandb.")
@click.option("--wandb-project", default="meaningbert-v2-comparatifs", help="Project the comparison is published to.")
def main(
    source: str,
    runs_dir: str,
    entity: str,
    project_prefix: str,
    figures_dir: str,
    tables_dir: str,
    html_out: str,
    wandb_upload: bool,
    wandb_project: str,
) -> None:
    """Build every comparison artefact from whichever source is asked for."""
    set_theme()
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    runs = load_runs(runs_dir) if source == "local" else load_from_wandb(entity, project_prefix)
    frame = runs_to_frame(runs)
    if frame.empty:
        raise SystemExit(f"no run found in {runs_dir if source == 'local' else project_prefix}")

    print(f"{len(frame)} runs read from {source}")
    print(frame.groupby(["head", "arch"], observed=True).size().to_string())

    figures = [
        figure_head_effect(frame, os.path.join(figures_dir, "v2-tete-de-sortie.pdf")),
        figure_objective_plane(frame, os.path.join(figures_dir, "v2-plan-objectif.pdf")),
        figure_grid_heatmap(frame, os.path.join(figures_dir, "v2-grille.pdf")),
        figure_corpus_and_augmentation(frame, os.path.join(figures_dir, "v2-corpus-augmentation.pdf")),
    ]
    figures = [path for path in figures if path]
    tables = [
        table_head_effect(frame, os.path.join(tables_dir, "v2-tete-de-sortie.tex")),
        table_grid(frame, os.path.join(tables_dir, "v2-grille.tex")),
        table_seeds(frame, os.path.join(tables_dir, "v2-graines.tex")),
    ]

    frame.to_csv(os.path.join(tables_dir, "v2-runs.csv"), index=False)
    grid_done = int(len(frame[(frame["head"] == "clamped") & frame["variant"].isin(GRID_VARIANTS)]))
    report = html_report(frame, figures_dir, html_out, done=grid_done, total=20)
    print(f"\nrapport : {report}")
    print("\nfigures :")
    for path in figures:
        print(f"  {path}")
    print("tables :")
    for path in [t for t in tables if t]:
        print(f"  {path}")

    if wandb_upload:
        url = upload_to_wandb(frame, figures, entity, wandb_project)
        print(f"\nwandb : {url}")


if __name__ == "__main__":
    main()
