import os

import click
import wandb

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
