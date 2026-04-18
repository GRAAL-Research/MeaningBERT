"""Analyze MeaningBERT checkpoint sweep results from wandb.

Pulls all finished runs from the wandb project, groups by checkpoint and
augmentation strategy, and produces summary tables (CSV + LaTeX) with
mean +/- std across folds.

Usage::

    python analyze_sweep_results.py --project davebulaval/meaningbert-checkpoint-sweep
    python analyze_sweep_results.py --project davebulaval/meaningbert-checkpoint-sweep --output_dir ./results
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from statistics import mean, stdev

import wandb

AUGMENTATION_LABELS: dict[str, str] = {
    "none": "No augmentation",
    "swap": "Swap",
    "back_translation": "Back-translation",
}

# Metrics to extract from wandb run summaries.
TEST_METRICS: list[str] = [
    "test/rmse",
    "test/R2",
    "test/pearson_corr",
    "test/pearson_pvalue",
    "test/mean_score",
    "test/st_dev_score",
]

HOLDOUT_IDENTICAL_METRICS: list[str] = [
    "train/test/identical_sentences_rmse",
    "train/test/identical_sentences_ratio_95",
    "train/test/identical_sentences_ratio_99",
    "train/test/identical_sentences_ratio_equals",
    "train/test/identical_sentences_mean_score",
]

HOLDOUT_UNRELATED_METRICS: list[str] = [
    "train/test/unrelated_sentences_rmse",
    "train/test/unrelated_sentences_ratio_1",
    "train/test/unrelated_sentences_ratio_5",
    "train/test/unrelated_sentences_ratio_equals",
    "train/test/unrelated_sentences_mean_score",
]


def fetch_runs(project: str) -> list[dict]:
    """Fetch all finished runs from wandb and return structured dicts."""
    api = wandb.Api()
    runs = api.runs(project, filters={"state": "finished"})

    results: list[dict] = []
    for run in runs:
        config = run.config
        summary = dict(run.summary)

        # Extract config fields logged by few_shot_training.py
        checkpoint = config.get("checkpoint", run.name.split("_seed")[0] if "_seed" in run.name else "unknown")
        augmentation = config.get("data_augmentation", "unknown")
        fold = config.get("fold")
        seed = config.get("seed") or _extract_seed_from_name(run.name)

        entry = {
            "name": run.name,
            "id": run.id,
            "created_at": run.created_at,
            "checkpoint": checkpoint,
            "augmentation": augmentation,
            "fold": fold,
            "seed": seed,
        }

        # Collect all metrics
        for metric_key in TEST_METRICS + HOLDOUT_IDENTICAL_METRICS + HOLDOUT_UNRELATED_METRICS:
            value = summary.get(metric_key)
            # Handle nested dicts (e.g. R2 can be {"r_squared": 0.5})
            if isinstance(value, dict):
                value = value.get("r_squared", None)
            # Ensure numeric or None
            if value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
            entry[metric_key] = value

        results.append(entry)

    return results


def _extract_seed_from_name(name: str) -> int | None:
    """Parse seed from run name like 'bert-base-uncased_seed42_...'."""
    for part in name.split("_"):
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    return None


def deduplicate_runs(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep one run per (checkpoint, augmentation, fold), preferring the most recent.

    Returns (deduplicated_runs, duplicate_runs).
    """
    seen: dict[tuple, dict] = {}
    duplicates: list[dict] = []

    for run in runs:
        key = (run["checkpoint"], run["augmentation"], run["fold"])
        if key not in seen:
            seen[key] = run
        else:
            # Keep the most recent, mark the older one as duplicate
            existing = seen[key]
            if run["created_at"] > existing["created_at"]:
                duplicates.append(existing)
                seen[key] = run
            else:
                duplicates.append(run)

    if duplicates:
        print(f"  Deduplicated: removed {len(duplicates)} duplicate runs (kept most recent per fold)")
        for dup in duplicates:
            print(f"    {dup['name']} (id={dup['id']})")

    return list(seen.values()), duplicates


def group_runs(
    runs: list[dict],
) -> dict[tuple[str, str], list[dict]]:
    """Group runs by (checkpoint, augmentation)."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        key = (run["checkpoint"], run["augmentation"])
        groups[key].append(run)
    return dict(groups)


def compute_summary_table(groups: dict[tuple[str, str], list[dict]]) -> list[dict]:
    """Compute mean +/- std for each (checkpoint, augmentation) group."""
    metrics_to_summarize = [
        ("test/rmse", "RMSE", "low"),
        ("test/R2", "R2", "high"),
        ("test/pearson_corr", "Pearson", "high"),
        ("train/test/identical_sentences_ratio_equals", "Identical =100%", "high"),
        ("train/test/identical_sentences_ratio_95", "Identical >95%", "high"),
        ("train/test/unrelated_sentences_ratio_equals", "Unrelated =0%", "high"),
        ("train/test/unrelated_sentences_ratio_5", "Unrelated <5%", "high"),
    ]

    rows: list[dict] = []
    for (checkpoint, augmentation), run_list in sorted(groups.items()):
        row: dict = {
            "Checkpoint": checkpoint,
            "Augmentation": AUGMENTATION_LABELS.get(augmentation, augmentation),
            "N_folds": len(run_list),
        }

        for metric_key, label, _ in metrics_to_summarize:
            import math  # pylint: disable=import-outside-toplevel
            values = [
                r[metric_key]
                for r in run_list
                if isinstance(r.get(metric_key), float) and not math.isnan(r[metric_key])
            ]
            if len(values) >= 2:
                row[f"{label}_mean"] = mean(values)
                row[f"{label}_std"] = stdev(values)
                row[label] = f"{mean(values):.3f} +/- {stdev(values):.3f}"
            elif len(values) == 1:
                row[f"{label}_mean"] = values[0]
                row[f"{label}_std"] = 0.0
                row[label] = f"{values[0]:.3f}"
            else:
                row[f"{label}_mean"] = None
                row[f"{label}_std"] = None
                row[label] = "n/a"

        rows.append(row)

    return rows


def print_summary(rows: list[dict]) -> None:
    """Print a formatted summary table to stdout."""
    display_cols = [
        "Checkpoint",
        "Augmentation",
        "N_folds",
        "RMSE",
        "R2",
        "Pearson",
        "Identical =100%",
        "Identical >95%",
        "Unrelated =0%",
        "Unrelated <5%",
    ]

    # Header
    header = " | ".join(f"{col:>20s}" for col in display_cols)
    print(header)
    print("-" * len(header))

    for row in rows:
        line = " | ".join(f"{str(row.get(col, 'n/a')):>20s}" for col in display_cols)
        print(line)


def save_csv(rows: list[dict], output_path: str) -> None:
    """Save summary rows to CSV."""
    import csv  # pylint: disable=import-outside-toplevel

    cols = [
        "Checkpoint",
        "Augmentation",
        "N_folds",
        "RMSE",
        "R2",
        "Pearson",
        "Identical =100%",
        "Identical >95%",
        "Unrelated =0%",
        "Unrelated <5%",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved to {output_path}")


def generate_latex_table(rows: list[dict], output_path: str) -> None:
    """Generate a LaTeX table from summary rows."""
    metric_cols = [
        ("RMSE", "low"),
        ("R2", "high"),
        ("Pearson", "high"),
        ("Identical =100%", "high"),
        ("Unrelated =0%", "high"),
    ]

    # Find best values per metric
    best: dict[str, float | None] = {}
    for col, direction in metric_cols:
        means = [r.get(f"{col}_mean") for r in rows if r.get(f"{col}_mean") is not None]
        if means:
            best[col] = min(means) if direction == "low" else max(means)

    n_cols = 3 + len(metric_cols)  # checkpoint, aug, n_folds + metrics
    alignment = "l l c " + " ".join(["c"] * len(metric_cols))

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Checkpoint sweep results (mean $\pm$ std across folds).}")
    lines.append(r"\label{tab:checkpoint-sweep}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{" + alignment + "}")
    lines.append(r"\toprule")

    # Header
    header_parts = ["Checkpoint", "Augmentation", "Folds"]
    header_parts += [col for col, _ in metric_cols]
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    prev_checkpoint = None
    for row in rows:
        checkpoint = row["Checkpoint"]
        if prev_checkpoint is not None and checkpoint != prev_checkpoint:
            lines.append(r"\midrule")
        prev_checkpoint = checkpoint

        parts = [
            checkpoint.replace("_", r"\_"),
            row["Augmentation"],
            str(row["N_folds"]),
        ]

        for col, _ in metric_cols:
            val_str = row.get(col, "n/a")
            mean_val = row.get(f"{col}_mean")
            if mean_val is not None and best.get(col) is not None and abs(mean_val - best[col]) < 1e-6:
                val_str = r"\textbf{" + val_str + "}"
            parts.append(val_str)

        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"LaTeX table saved to {output_path}")


def print_progress_report(project: str) -> None:
    """Print a quick status report of running vs finished runs."""
    api = wandb.Api()
    runs = api.runs(project)

    status_counts: dict[str, int] = defaultdict(int)
    checkpoint_progress: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for run in runs:
        state = run.state
        status_counts[state] += 1

        checkpoint = run.config.get("checkpoint", "unknown")
        checkpoint_progress[checkpoint][state] += 1

    print("=== Sweep Progress ===")
    for state, count in sorted(status_counts.items()):
        print(f"  {state}: {count}")

    print("\n=== Per Checkpoint ===")
    for checkpoint, states in sorted(checkpoint_progress.items()):
        parts = ", ".join(f"{s}={c}" for s, c in sorted(states.items()))
        print(f"  {checkpoint}: {parts}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Analyze MeaningBERT sweep results from wandb.")
    parser.add_argument(
        "--project",
        type=str,
        default="davebulaval/meaningbert-checkpoint-sweep",
        help="Wandb project path (entity/project).",
    )
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory for output files.")
    parser.add_argument(
        "--progress_only",
        action="store_true",
        help="Only print progress report, don't generate tables.",
    )
    args = parser.parse_args()

    print_progress_report(args.project)

    if args.progress_only:
        return

    runs = fetch_runs(args.project)
    if not runs:
        print("\nNo finished runs found. Tables will be generated once runs complete.")
        return

    runs, _ = deduplicate_runs(runs)
    groups = group_runs(runs)
    rows = compute_summary_table(groups)

    print(f"\n=== Summary ({len(runs)} finished runs) ===\n")
    print_summary(rows)

    os.makedirs(args.output_dir, exist_ok=True)
    save_csv(rows, os.path.join(args.output_dir, "sweep_summary.csv"))
    generate_latex_table(rows, os.path.join(args.output_dir, "sweep_summary.tex"))


if __name__ == "__main__":
    main()
