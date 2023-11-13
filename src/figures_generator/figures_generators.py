from statistics import mean, stdev
from typing import List

from python2latex import Document, Table, italic

all_metrics = [
    "FKGL",
    "SARI",
    "BLEU",
    "BLEU-SARI (AM)",
    "BLEU-SARI (GM)",
    "BLEURT",
    "IBLEU",
    "FKBLEU",
    "METEOR",
    "BERTScore precision",
    "BERTScore recall",
    "BERTScore F1",
    "QuestEval",
    "LENS",
    "Coverage",
    "Sentence Transformer",
    "TER",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
]
all_metrics.sort()

subset_metrics = [
    "SARI",
    "BLEU",
    "BLEU-SARI (AM)",
    "BLEU-SARI (GM)",
    "BLEURT",
    "IBLEU",
    "FKBLEU",
    "METEOR",
    "BERTScore precision",
    "BERTScore recall",
    "BERTScore F1",
    "QuestEval",
    "LENS",
    "Coverage",
    "Sentence Transformer",
    "TER",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
]
subset_metrics.sort()


def get_table_1115(few_shot_data: List, saving_dir: str):
    alpha = 0.999
    doc = Document(filename="res_1115", filepath=saving_dir, doc_type="article", border="10pt")

    # Create the data
    col, row = 4, 23

    table = doc.new(
        Table(
            shape=(row + 1, col),
            as_float_env=True,
            alignment=["l"] + ["c"] * (col - 1),
            caption=r"Results of the benchmarking metrics and MeaningBERT trained without data augmentation (DA) "
            r"and with DA. \textbf{Bolded} value are the best results and \textit{italic} one are results "
            r"with a p-value $\alpha \leq 0.999$.",
            caption_pos="bottom",
        )
    )

    table[0, 0:] = [
        "Metrics",
        "Pearson",
        "R$^2$",
        "RMSE",
    ]
    table[0, 0:].add_rule()

    # Pearson
    col_idx = 1
    data = []
    for metric in all_metrics:
        metric_data = []
        for run_score in few_shot_data[3]:
            metric_data.append(run_score[f"test/{metric}_pearson_corr"])

        data.append((metric, metric_data))

    for idx, (metric_name, metric_values) in enumerate(data):
        # Row, column
        table[idx + 1, 0] = metric_name

        # Pearson is column 1
        table[idx + 1, col_idx] = mean(metric_values)

    # We add few-shot data results
    table[-2, 0] = "MeaningBERT (without DA)"
    table[-1, 0] = "MeaningBERT (with DA)"
    table[-3, 0:].add_rule()

    no_data_augmentation = []
    for run_score in few_shot_data[0]:  # No DA
        no_data_augmentation.append(run_score["test/pearson_corr"])
    table[-2, col_idx] = mean(no_data_augmentation)

    data_augmentation = []
    for run_score in few_shot_data[1]:  # With DA
        data_augmentation.append(run_score["test/pearson_corr"])
    table[-1, col_idx] = mean(data_augmentation)

    table[:, col_idx].highlight_best("high", "bold")

    for idx, (_, metric_values) in enumerate(data):
        table[idx + 1, col_idx] = (
            f"{round(table[idx + 1, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(metric_values), 2)}"
        )

    data_p_value = []
    for metric in all_metrics:
        metric_data = []
        for run_score in few_shot_data[3]:
            metric_data.append(run_score[f"test/{metric}_pearson_pvalue"])

        data_p_value.append((metric, metric_data))

    for idx, (_, metric_values) in enumerate(data_p_value):
        if mean(metric_values) <= (1 - alpha):
            table[idx + 1, col_idx].apply_command(lambda value: italic(value))

    no_data_augmentation_p_value = []
    for run_score in few_shot_data[0]:  # No DA
        no_data_augmentation_p_value.append(run_score["test/pearson_pvalue"])
    if mean(no_data_augmentation_p_value) <= (1 - alpha):
        table[-2, col_idx].apply_command(lambda value: italic(value))

    data_augmentation_p_value = []
    for run_score in few_shot_data[1]:  # With DA
        data_augmentation_p_value.append(run_score["test/pearson_pvalue"])
    if mean(data_augmentation_p_value) <= (1 - alpha):
        table[-1:, col_idx].apply_command(lambda value: italic(value))

    # We add +/-
    table[-2, col_idx] = (
        f"{round(table[-2, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(no_data_augmentation), 2)}"
    )
    table[-1:, col_idx] = (
        f"{round(table[-1:, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(data_augmentation), 2)}"
    )

    # r_squared
    col_idx = 2
    data = []
    for metric in all_metrics:
        metric_data = []
        for run_score in few_shot_data[3]:
            metric_data.append(run_score[f"test/{metric}_R2"])

        data.append((metric, metric_data))

    for idx, (_, metric_values) in enumerate(data):
        table[idx + 1, col_idx] = mean(metric_values)

    no_data_augmentation = []
    for run_score in few_shot_data[0]:  # No DA
        no_data_augmentation.append(run_score["test/R2"])
    table[-2, col_idx] = mean(no_data_augmentation)

    data_augmentation = []
    for run_score in few_shot_data[1]:  # With DA
        data_augmentation.append(run_score["test/R2"])
    table[-1:, col_idx] = mean(data_augmentation)

    table[:, col_idx].highlight_best("high", "bold")

    for idx, (_, metric_values) in enumerate(data):
        table[idx + 1, col_idx] = (
            f"{round(table[idx + 1, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(metric_values), 2)}"
        )

    # We add +/-
    table[-2, col_idx] = (
        f"{round(table[-2, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(no_data_augmentation), 2)}"
    )
    table[-1:, col_idx] = (
        f"{round(table[-1:, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(data_augmentation), 2)}"
    )

    # RMSE
    col_idx = 3
    data = []
    for metric in all_metrics:
        metric_data = []
        for run_score in few_shot_data[3]:
            metric_data.append(run_score[f"test/{metric}_rmse"])

        data.append((metric, metric_data))

    for idx, (_, metric_values) in enumerate(data):
        table[idx + 1, col_idx] = mean(metric_values)

    no_data_augmentation = []
    for run_score in few_shot_data[0]:  # No DA
        no_data_augmentation.append(run_score["test/rmse"])
    table[-2, col_idx] = mean(no_data_augmentation)

    data_augmentation = []
    for run_score in few_shot_data[1]:  # With DA
        data_augmentation.append(run_score["test/rmse"])
    table[-1:, col_idx] = mean(data_augmentation)

    table[:, col_idx].highlight_best("low", "bold")

    for idx, (_, metric_values) in enumerate(data):
        table[idx + 1, col_idx] = (
            f"{round(table[idx + 1, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(metric_values), 2)}"
        )

    # We add +/-
    table[-2, col_idx] = (
        f"{round(table[-2, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(no_data_augmentation), 2)}"
    )
    table[-1:, col_idx] = (
        f"{round(table[-1:, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(data_augmentation), 2)}"
    )

    return doc


def get_table_112(few_shot_data: List, saving_dir: str, count_equals=False):
    doc = Document(filename="res_112", filepath=saving_dir, doc_type="article", border="10pt")

    if count_equals:
        col = 4
        headers = [
            "Metrics",
            "\% greater than 95\%",
            "\% greater than 99\%",
            "\% equal to 100\%",
        ]
    else:
        col = 3
        headers = [
            "Metrics",
            "\% greater than 95\%",
            "\% greater than 99\%",
        ]
    row = 22

    table = doc.new(
        Table(
            shape=(row + 1, col),
            as_float_env=True,
            alignment=["l"] + ["c"] * (col - 1),
            caption=r"Percentages of time a metric returns the expected rating for the unrelated sentence test using "
            r"the sanity check dataset same sentence split. \textbf{Bolded} value are the best results.",
            caption_pos="bottom",
            label="tab:112",
        )
    )

    table[0, 0:] = headers
    table[0, 0:].add_rule()

    # Count 95\%
    col_idx = 1
    data = []
    for metric in subset_metrics:
        metric_data = []
        for run_score in few_shot_data[3]:
            metric_data.append(run_score[f"test/same_sentence_{metric}_ratio_95"])

        data.append((metric, metric_data))

    for idx, (metric_name, metric_values) in enumerate(data):
        # Row, column
        table[idx + 1, 0] = metric_name

        table[idx + 1, col_idx] = mean(metric_values)

    # We add few-shot data results
    table[-2, 0] = "MeaningBERT (without DA)"
    table[-1, 0] = "MeaningBERT (with DA)"
    table[-3, 0:].add_rule()

    no_data_augmentation = []
    for run_score in few_shot_data[3]:  # No DA
        no_data_augmentation.append(run_score["train/test/same_sentence_ratio_95"])
    table[-2, col_idx] = mean(no_data_augmentation)

    data_augmentation = []
    for run_score in few_shot_data[4]:  # With DA
        data_augmentation.append(run_score["train/test/same_sentence_ratio_95"])
    table[-1, col_idx] = mean(data_augmentation)

    table[:, col_idx].highlight_best("high", "bold")

    for idx, (_, metric_values) in enumerate(data):
        table[idx + 1, col_idx] = (
            f"{round(table[idx + 1, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(metric_values), 2)}"
        )

    # We add +/-
    table[-2, col_idx] = (
        f"{round(table[-2, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(no_data_augmentation), 2)}"
    )
    table[-1:, col_idx] = (
        f"{round(table[-1:, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(data_augmentation), 2)}"
    )

    # Count 99\%
    col_idx = 2
    data = []
    for metric in subset_metrics:
        metric_data = []
        for run_score in few_shot_data[3]:
            metric_data.append(run_score[f"test/same_sentence_{metric}_ratio_99"])

        data.append((metric, metric_data))

    for idx, (_, metric_values) in enumerate(data):
        table[idx + 1, col_idx] = mean(metric_values)

    no_data_augmentation = []
    for run_score in few_shot_data[3]:  # No DA
        no_data_augmentation.append(run_score["train/test/same_sentence_ratio_99"])
    table[-2, col_idx] = mean(no_data_augmentation)

    data_augmentation = []
    for run_score in few_shot_data[4]:  # With DA
        data_augmentation.append(run_score["train/test/same_sentence_ratio_99"])
    table[-1, col_idx] = mean(data_augmentation)

    table[:, col_idx].highlight_best("high", "bold")

    for idx, (_, metric_values) in enumerate(data):
        table[idx + 1, col_idx] = (
            f"{round(table[idx + 1, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(metric_values), 2)}"
        )

    # We add +/-
    table[-2, col_idx] = (
        f"{round(table[-2, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(no_data_augmentation), 2)}"
    )
    table[-1:, col_idx] = (
        f"{round(table[-1:, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(data_augmentation), 2)}"
    )

    if count_equals:
        # Count 100\%
        col_idx = 3
        data = []
        for metric in subset_metrics:
            metric_data = []
            for run_score in few_shot_data[3]:
                metric_data.append(run_score[f"test/same_sentence_{metric}_ratio_equals"])

            data.append((metric, metric_data))

        for idx, (_, metric_values) in enumerate(data):
            table[idx + 1, col_idx] = mean(metric_values)

        no_data_augmentation = []
        for run_score in few_shot_data[3]:  # No DA
            no_data_augmentation.append(run_score["train/test/same_sentence_ratio_equals"])
        table[-2, col_idx] = mean(no_data_augmentation)

        data_augmentation = []
        for run_score in few_shot_data[4]:  # With DA
            data_augmentation.append(run_score["train/test/same_sentence_ratio_equals"])
        table[-1, col_idx] = mean(data_augmentation)

        table[:, col_idx].highlight_best("high", "bold")

        for idx, (_, metric_values) in enumerate(data):
            table[idx + 1, col_idx] = (
                f"{round(table[idx + 1, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(metric_values), 2)}"
            )

        # We add +/-
        table[-2, col_idx] = (
            f"{round(table[-2, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(no_data_augmentation), 2)}"
        )
        table[-1:, col_idx] = (
            f"{round(table[-1:, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(data_augmentation), 2)}"
        )
    return doc


def get_table_1131(few_shot_data: List, saving_dir: str, count_equals=False):
    doc = Document(filename="res_1131", filepath=saving_dir, doc_type="article", border="10pt")

    # Create the data
    if count_equals:
        col = 4
        headers = [
            "Metrics",
            "\% greater than 5\%",
            "\% greater than 1\%",
            "\% equal to 0\%",
        ]
    else:
        col = 3
        headers = [
            "Metrics",
            "\% greater than 5\%",
            "\% greater than 1\%",
        ]
    row = 22

    table = doc.new(
        Table(
            shape=(row + 1, col),
            as_float_env=True,
            alignment=["l"] + ["c"] * (col - 1),
            caption=r"Percentages of time a metric returns the expected rating for the unrelated sentence test using "
            r"the sanity check dataset unrelated sentence split. \textbf{Bolded} value are the best results.",
            caption_pos="bottom",
        )
    )

    table[0, 0:] = headers
    table[0, 0:].add_rule()

    # Count 5\%
    col_idx = 1
    data = []
    for metric in subset_metrics:
        metric_data = []
        for run_score in few_shot_data[3]:
            metric_data.append(run_score[f"test/irrelevant_sentence_{metric}_ratio_5"])

        data.append((metric, metric_data))

    for idx, (metric_name, metric_values) in enumerate(data):
        # Row, column
        table[idx + 1, 0] = metric_name

        table[idx + 1, col_idx] = mean(metric_values)

    # We add few-shot data results
    table[-2, 0] = "MeaningBERT (without DA)"
    table[-1, 0] = "MeaningBERT (with DA)"
    table[-3, 0:].add_rule()

    no_data_augmentation = []
    for run_score in few_shot_data[3]:  # No DA
        no_data_augmentation.append(run_score["train/test/irrelevant_sentence_ratio_5"])
    table[-2, col_idx] = mean(no_data_augmentation)

    data_augmentation = []
    for run_score in few_shot_data[4]:  # With DA
        data_augmentation.append(run_score["train/test/irrelevant_sentence_ratio_5"])
    table[-1, col_idx] = mean(data_augmentation)

    table[:, col_idx].highlight_best("high", "bold")

    for idx, (_, metric_values) in enumerate(data):
        table[idx + 1, col_idx] = (
            f"{round(table[idx + 1, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(metric_values), 2)}"
        )

    # We add +/-
    table[-2, col_idx] = (
        f"{round(table[-2, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(no_data_augmentation), 2)}"
    )
    table[-1:, col_idx] = (
        f"{round(table[-1:, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(data_augmentation), 2)}"
    )

    # Count 1\%
    col_idx = 2
    data = []
    for metric in subset_metrics:
        metric_data = []
        for run_score in few_shot_data[3]:
            metric_data.append(run_score[f"test/irrelevant_sentence_{metric}_ratio_1"])

        data.append((metric, metric_data))

    for idx, (_, metric_values) in enumerate(data):
        table[idx + 1, col_idx] = mean(metric_values)

    no_data_augmentation = []
    for run_score in few_shot_data[3]:  # No DA
        no_data_augmentation.append(run_score["train/test/irrelevant_sentence_ratio_1"])
    table[-2, col_idx] = mean(no_data_augmentation)

    data_augmentation = []
    for run_score in few_shot_data[4]:  # With DA
        data_augmentation.append(run_score["train/test/irrelevant_sentence_ratio_1"])
    table[-1, col_idx] = mean(data_augmentation)

    table[:, col_idx].highlight_best("high", "bold")

    for idx, (_, metric_values) in enumerate(data):
        table[idx + 1, col_idx] = (
            f"{round(table[idx + 1, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(metric_values), 2)}"
        )

    # We add +/-
    table[-2, col_idx] = (
        f"{round(table[-2, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(no_data_augmentation), 2)}"
    )
    table[-1:, col_idx] = (
        f"{round(table[-1:, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(data_augmentation), 2)}"
    )

    # Count 0\%
    if count_equals:
        col_idx = 3
        data = []
        for metric in subset_metrics:
            metric_data = []
            for run_score in few_shot_data[3]:
                metric_data.append(run_score[f"test/irrelevant_sentence_{metric}_ratio_equals"])

            data.append((metric, metric_data))

        for idx, (_, metric_values) in enumerate(data):
            table[idx + 1, col_idx] = mean(metric_values)

        no_data_augmentation = []
        for run_score in few_shot_data[3]:  # No DA
            no_data_augmentation.append(run_score["train/test/irrelevant_sentence_ratio_equals"])
        table[-2, col_idx] = mean(no_data_augmentation)

        data_augmentation = []
        for run_score in few_shot_data[4]:  # With DA
            data_augmentation.append(run_score["train/test/irrelevant_sentence_ratio_equals"])
        table[-1, col_idx] = mean(data_augmentation)

        table[:, col_idx].highlight_best("high", "bold")

        for idx, (_, metric_values) in enumerate(data):
            table[idx + 1, col_idx] = (
                f"{round(table[idx + 1, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(metric_values), 2)}"
            )

        # We add +/-
        table[-2, col_idx] = (
            f"{round(table[-2, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(no_data_augmentation), 2)}"
        )
        table[-1:, col_idx] = (
            f"{round(table[-1:, col_idx].data[0][0], 3)}" + r"$\pm$" + f"{round(stdev(data_augmentation), 2)}"
        )
    return doc
