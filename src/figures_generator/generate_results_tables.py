"""Build the LaTeX result tables of the article from a wandb project.

Runs flagged ``diverged`` by ``metrics.compute_metrics``, and the runs of the previous
sweep that show the same collapse signature, are excluded before any averaging. See
``docs/H1-diagnostic-calibration.md``, correction C1.
"""

import os

import click
import wandb

from compression_report import drop_unusable_runs
from figures_generators import (
    get_table_112,
    get_table_1115,
    get_table_1131,
)


@click.command()
@click.argument("wandb_project_name")
@click.argument("figures_saving_directory")
def correction(wandb_project_name, figures_saving_directory):
    os.makedirs(figures_saving_directory, exist_ok=True)
    api = wandb.Api()

    runs = api.runs(wandb_project_name)

    data_augmentation_true = []
    data_augmentation_false = []

    previous_scores = []
    holdout_500_true = []
    holdout_500_false = []
    for run in runs:
        if run.state == "finished":
            data_augmentation = run.config.get("data_augmentation")

            if data_augmentation == "True 250 holdout fixed":
                data_augmentation_true.append(dict(run.summary))
            elif data_augmentation == "False 250 holdout fixed":
                data_augmentation_false.append(dict(run.summary))
            elif data_augmentation == "False 250 holdout":
                previous_scores.append(dict(run.summary))
            elif data_augmentation == "True 500 holdout fixed":
                holdout_500_true.append(dict(run.summary))
            elif data_augmentation == "False 500 holdout fixed":
                holdout_500_false.append(dict(run.summary))

    # C1: a diverged run reports NaN metrics and must not enter a mean.
    data_augmentation_true = drop_unusable_runs(data_augmentation_true, "data_augmentation_true")
    data_augmentation_false = drop_unusable_runs(data_augmentation_false, "data_augmentation_false")
    previous_scores = drop_unusable_runs(previous_scores, "previous_scores")
    holdout_500_true = drop_unusable_runs(holdout_500_true, "holdout_500_true")
    holdout_500_false = drop_unusable_runs(holdout_500_false, "holdout_500_false")

    required = {
        "holdout_500_false": holdout_500_false,
        "holdout_500_true": holdout_500_true,
        "data_augmentation_false": data_augmentation_false,
        "data_augmentation_true": data_augmentation_true,
    }
    empty = [name for name, lst in required.items() if not lst]
    if empty:
        raise SystemExit(f"No finished runs found for: {', '.join(empty)}. Check the wandb project name.")

    doc_1115 = get_table_1115(
        few_shot_data=[
            data_augmentation_false,
            data_augmentation_true,
            previous_scores,
            holdout_500_false,
            holdout_500_true,
        ],
        saving_dir=figures_saving_directory,
    )
    doc_1115.build()

    doc_112 = get_table_112(
        few_shot_data=[
            data_augmentation_false,
            data_augmentation_true,
            previous_scores,
            holdout_500_false,
            holdout_500_true,
        ],
        saving_dir=figures_saving_directory,
    )
    doc_112.build()

    doc_1131 = get_table_1131(
        few_shot_data=[
            data_augmentation_false,
            data_augmentation_true,
            previous_scores,
            holdout_500_false,
            holdout_500_true,
        ],
        saving_dir=figures_saving_directory,
    )
    doc_1131.build()


if __name__ == "__main__":
    correction()
