import os.path
import statistics
import urllib
from statistics import mean, geometric_mean, stdev
from zipfile import ZipFile

import torch
from evaluate import load
from lens import LENS
from questeval.questeval_metric import QuestEval
from sklearn.metrics import mean_squared_error
from textstat import textstat
from tqdm import tqdm

from metrics.implemented_metrics import (
    SentenceBertWrapper,
    IBLEU,
    FKBLEU,
)
from metrics.model_salience import CoverageModel

# To handle LENS deterministic pattern fixed
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

torch.multiprocessing.set_sharing_strategy("file_system")

r2_metric = load("r_squared")
pearsonr_metric = load("pearsonr")

sari = load("sari")

# We use SacreBLEU that produce more reproducible results
bleu = load("sacrebleu")

# We use the largest, and supposedly better BLEURT model
bleurt = load("bleurt", module_type="metric", checkpoint="bleurt-large-512")

ibleu = IBLEU(alpha=0.9)

fkbleu = FKBLEU(alpha=0.9)

meteor = load("meteor")

bertscore = load("bertscore")

questeval = QuestEval(
    task="text2text",
    language="en",
    no_cuda=False,
    do_weighter=False,
    list_scores=["bertscore", "answerability"],
    qg_batch_size=36,
)

lens_model_ckpt = os.path.join("LENS", "checkpoints", "lens_model.ckpt")
if not os.path.exists(lens_model_ckpt):
    url = "https://github.com/GRAAL-Research/MeaningBERT/releases/download/dependencies_model_release/LENS.zip"
    filehandle, _ = urllib.request.urlretrieve(url)
    with ZipFile(".", "r") as zip_file:
        zip_file.extractall(lens_model_ckpt, members=None, pwd=None)
lens = LENS(lens_model_ckpt, rescale=True)

coverage_model_ckpt = os.path.join("coverage_roberta.bin")
if not os.path.exists(coverage_model_ckpt):
    url = (
        "https://github.com/GRAAL-Research/MeaningBERT/releases/download/dependencies_model_release/"
        "coverage_roberta.bin"
    )
    filehandle, _ = urllib.request.urlretrieve(url)
    with ZipFile(".", "r") as zip_file:
        zip_file.extractall(coverage_model_ckpt, members=None, pwd=None)

coverage_kis = CoverageModel(model_file=coverage_model_ckpt)

# For Semantic Textual Similarity
sentence_transformer = SentenceBertWrapper("all-roberta-large-v1")

ter = load("ter")

rouge = load("rouge")


def compute_other_metrics_performance(
    test_set, holdout_identical_set, holdout_unrelated_set, logger, device: str = "cuda:0"
):
    compute_other_metrics_on_test_set(test_set=test_set, logger=logger, device=device)
    compute_other_metrics_on_holdout_identical_set(
        holdout_identical_set=holdout_identical_set, logger=logger, device=device
    )

    compute_other_metrics_on_holdout_unrelated_set(
        holdout_unrelated_set=holdout_unrelated_set, logger=logger, device=device
    )


def compute_other_metrics_on_test_set(test_set, logger, device: str = "cuda:0"):
    fkgl_scores = []
    sari_scores = []
    bleu_scores = []
    bleu_sari_am_scores = []
    bleu_sari_gm_scores = []
    bleurt_scores = []
    ibleu_scores = []
    fkbleu_scores = []
    meteor_scores = []
    bertscore_precision_scores = []
    bertscore_recall_scores = []
    bertscore_f1_scores = []
    questeval_scores = []
    lens_scores = []
    coverage_scores = []
    sentence_transformer_scores = []
    ter_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    ratings = []

    # Processing of all scores on the dataset
    for eval_data in tqdm(test_set, desc="Doing test evaluation on best selected metrics:"):
        original = [eval_data["original"]]
        simplification = [eval_data["simplification"]]
        rating = eval_data["label"]
        ratings.append(rating)

        # Between 0 and 100: 100 is the best
        fkgl_scores.append(textstat.flesch_kincaid_grade(simplification[0]))

        # Between 0 and 100: 100 is the best
        sari_score = sari.compute(sources=original, predictions=simplification, references=[original])["sari"]
        sari_scores.append(sari_score)

        # Between 0 and 100: 100 is the best
        bleu_score = bleu.compute(predictions=simplification, references=[original])["score"]
        bleu_scores.append(bleu_score)

        bleu_sari_am_scores.append(mean([sari_score, bleu_score]))
        try:
            bleu_sari_gm_scores.append(geometric_mean([sari_score, bleu_score]))
        except statistics.StatisticsError:
            bleu_sari_gm_scores.append(0)

        # Between 0 and 100: 100 is the best
        bleurt_scores.append(bleurt.compute(predictions=simplification, references=original)["scores"][0] * 100)

        ibleu_scores.append(ibleu.compute(source=original, prediction=simplification, reference=original))

        fkbleu_scores.append(fkbleu.compute(source=original, prediction=simplification, reference=original))

        # Between 0 and 100: 100 is the best
        meteor_scores.append(meteor.compute(predictions=simplification, references=original)["meteor"] * 100)

        # Default model is the best
        # We use the precision, recall and F1
        # Between 0 and 100: 100 is the best
        bertscore_score = bertscore.compute(predictions=simplification, references=original, lang="en", device=device)
        bertscore_precision_scores.append(bertscore_score["precision"][0] * 100)
        bertscore_recall_scores.append(bertscore_score["recall"][0] * 100)
        bertscore_f1_scores.append(bertscore_score["f1"][0] * 100)

        # Between 0 and 100: 100 is the best
        questeval_scores.append(
            questeval.corpus_questeval(sources=original, hypothesis=simplification)["ex_level_scores"][0] * 100
        )

        # Between 0 and 100: 100 is the best
        lens_scores.append(
            lens.score(
                complex=original,
                simplified=simplification,
                references=[original],
                batch_size=1,
                gpus=1,
            )[0]
        )

        # Between 0 and 100: 100 is the best
        coverage_scores.append(coverage_kis.score(bodies=original, decodeds=simplification)["scores"][0] * 100)

        # Between o and 100: 100 is the best
        sentence_transformer_scores.append(
            sentence_transformer.compute(prediction=simplification[0], reference=original[0]).detach().cpu().item()
            * 100
        )

        ter_scores.append(100 - ter.compute(predictions=simplification, references=[original])["score"])

        rouge_score = rouge.compute(predictions=simplification, references=original)
        rouge_1_scores.append(rouge_score["rouge1"] * 100)
        rouge_2_scores.append(rouge_score["rouge2"] * 100)
        rouge_l_scores.append(rouge_score["rougeL"] * 100)

    # Compilation of the scores per metrics
    for metric_name, metric_score in zip(
        [
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
        ],
        [
            fkgl_scores,
            sari_scores,
            bleu_scores,
            bleu_sari_am_scores,
            bleu_sari_gm_scores,
            bleurt_scores,
            ibleu_scores,
            fkbleu_scores,
            meteor_scores,
            bertscore_precision_scores,
            bertscore_recall_scores,
            bertscore_f1_scores,
            questeval_scores,
            lens_scores,
            coverage_scores,
            sentence_transformer_scores,
            ter_scores,
            rouge_1_scores,
            rouge_2_scores,
            rouge_l_scores,
        ],
    ):
        pearson_corr = pearsonr_metric.compute(predictions=metric_score, references=ratings, return_pvalue=True)
        logger.log({f"test/{metric_name}_pearson_corr": pearson_corr["pearsonr"]})
        logger.log({f"test/{metric_name}_pearson_pvalue": pearson_corr["p-value"]})

        r_squared = r2_metric.compute(predictions=metric_score, references=ratings)
        logger.log({f"test/{metric_name}_R2": r_squared})

        rmse_score = mean_squared_error(metric_score, ratings, squared=False)
        logger.log({f"test/{metric_name}_rmse": rmse_score})

        logger.log({f"test/{metric_name}_mean": mean(metric_score)})
        logger.log({f"test/{metric_name}_st_dev": stdev(metric_score)})


def compute_other_metrics_on_holdout_identical_set(holdout_identical_set, logger, device: str = "cuda:0"):
    sari_scores = []
    bleu_scores = []
    bleu_sari_am_scores = []
    bleu_sari_gm_scores = []
    bleurt_scores = []
    ibleu_scores = []
    fkbleu_scores = []
    meteor_scores = []
    bertscore_precision_scores = []
    bertscore_recall_scores = []
    bertscore_f1_scores = []
    questeval_scores = []
    lens_scores = []
    coverage_scores = []
    sentence_transformer_scores = []
    ter_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    ratings = []

    # Processing of all scores on the dataset
    for eval_data in tqdm(
        holdout_identical_set,
        desc="Doing holdout evaluation on identical on best selected metrics:",
    ):
        original = [eval_data["original"]]
        simplification = [eval_data["simplification"]]
        rating = eval_data["label"]
        ratings.append(rating)
        # between 0 and 100: 100 is the best
        sari_score = sari.compute(sources=original, predictions=simplification, references=[original])["sari"]
        sari_scores.append(sari_score)

        # between 0 and 100: 100 is the best
        bleu_score = bleu.compute(predictions=simplification, references=[original])["score"]
        bleu_scores.append(bleu_score)

        bleu_sari_am_scores.append(mean([sari_score, bleu_score]))
        try:
            bleu_sari_gm_scores.append(geometric_mean([sari_score, bleu_score]))
        except statistics.StatisticsError:
            bleu_sari_gm_scores.append(0)

        # between 0 and 100: 100 is the best
        bleurt_scores.append(bleurt.compute(predictions=simplification, references=original)["scores"][0] * 100)

        ibleu_scores.append(ibleu.compute(source=original, prediction=simplification, reference=original))

        fkbleu_scores.append(fkbleu.compute(source=original, prediction=simplification, reference=original))

        # between 0 and 100: 100 is the best
        meteor_scores.append(meteor.compute(predictions=simplification, references=original)["meteor"] * 100)

        # Default model is the best
        # We use the precision, recall and F1
        # between 0 and 100: 100 is the best
        bertscore_score = bertscore.compute(predictions=simplification, references=original, lang="en", device=device)
        bertscore_precision_scores.append(bertscore_score["precision"][0] * 100)
        bertscore_recall_scores.append(bertscore_score["recall"][0] * 100)
        bertscore_f1_scores.append(bertscore_score["f1"][0] * 100)

        # between 0 and 100: 100 is the best
        questeval_scores.append(
            questeval.corpus_questeval(sources=original, hypothesis=simplification)["ex_level_scores"][0] * 100
        )

        # between 0 and 100: 100 is the best
        lens_scores.append(
            lens.score(
                complex=original,
                simplified=simplification,
                references=[original],
                batch_size=1,
                gpus=1,
            )[0]
        )

        # between 0 and 100: 100 is the best
        coverage_scores.append(coverage_kis.score(bodies=original, decodeds=simplification)["scores"][0] * 100)

        # between o and 100: 100 is the best
        sentence_transformer_scores.append(
            sentence_transformer.compute(prediction=simplification[0], reference=original[0]).detach().cpu().item()
            * 100
        )

        ter_scores.append(100 - ter.compute(predictions=simplification, references=[original])["score"])

        rouge_score = rouge.compute(predictions=simplification, references=original)
        rouge_1_scores.append(rouge_score["rouge1"] * 100)
        rouge_2_scores.append(rouge_score["rouge2"] * 100)
        rouge_l_scores.append(rouge_score["rougeL"] * 100)

    # Compilation of the scores per metrics
    for metric_name, metric_score in zip(
        [
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
        ],
        [
            sari_scores,
            bleu_scores,
            bleu_sari_am_scores,
            bleu_sari_gm_scores,
            bleurt_scores,
            ibleu_scores,
            fkbleu_scores,
            meteor_scores,
            bertscore_precision_scores,
            bertscore_recall_scores,
            bertscore_f1_scores,
            questeval_scores,
            lens_scores,
            coverage_scores,
            sentence_transformer_scores,
            ter_scores,
            rouge_1_scores,
            rouge_2_scores,
            rouge_l_scores,
        ],
    ):
        counts_95 = [round(s) > 95 for s in metric_score]
        ratio_95 = sum(counts_95) / len(counts_95) * 100
        counts_99 = [round(s) > 99 for s in metric_score]
        ratio_99 = sum(counts_99) / len(counts_99) * 100
        counts_equals = [round(s) == 100 for s in metric_score]
        ratio_equals = sum(counts_equals) / len(counts_equals) * 100

        logger.log({f"test/same_sentence_{metric_name}_ratio_equals": ratio_equals})
        logger.log({f"test/same_sentence_{metric_name}_ratio_99": ratio_99})
        logger.log({f"test/same_sentence_{metric_name}_ratio_95": ratio_95})

        rmse_score = mean_squared_error(metric_score, ratings, squared=False)
        logger.log({f"test/same_sentence_{metric_name}_rmse": rmse_score})

        logger.log({f"test/same_sentence_{metric_name}_mean": mean(metric_score)})
        logger.log({f"test/same_sentence_{metric_name}_st_dev": stdev(metric_score)})


def compute_other_metrics_on_holdout_unrelated_set(holdout_unrelated_set, logger, device: str = "cuda:0"):
    sari_scores = []
    bleu_scores = []
    bleu_sari_am_scores = []
    bleu_sari_gm_scores = []
    bleurt_scores = []
    ibleu_scores = []
    fkbleu_scores = []
    meteor_scores = []
    bertscore_precision_scores = []
    bertscore_recall_scores = []
    bertscore_f1_scores = []
    questeval_scores = []
    lens_scores = []
    coverage_scores = []
    sentence_transformer_scores = []
    ter_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    ratings = []

    # Processing of all scores on the dataset
    for eval_data in tqdm(
        holdout_unrelated_set,
        desc="Doing holdout evaluation on unrelated best selected metrics:",
    ):
        original = [eval_data["original"]]
        simplification = [eval_data["simplification"]]
        rating = eval_data["label"]
        ratings.append(rating)

        # between 0 and 100: 100 is the best
        sari_score = sari.compute(sources=original, predictions=simplification, references=[original])["sari"]
        sari_scores.append(sari_score)

        # between 0 and 100: 100 is the best
        bleu_score = bleu.compute(predictions=simplification, references=[original])["score"]
        bleu_scores.append(bleu_score)

        bleu_sari_am_scores.append(mean([sari_score, bleu_score]))
        try:
            bleu_sari_gm_scores.append(geometric_mean([sari_score, bleu_score]))
        except statistics.StatisticsError:
            bleu_sari_gm_scores.append(0)

        # between 0 and 100: 100 is the best
        bleurt_scores.append(bleurt.compute(predictions=simplification, references=original)["scores"][0] * 100)

        ibleu_scores.append(ibleu.compute(source=original, prediction=simplification, reference=original))

        fkbleu_scores.append(fkbleu.compute(source=original, prediction=simplification, reference=original))

        # between 0 and 100: 100 is the best
        meteor_scores.append(meteor.compute(predictions=simplification, references=original)["meteor"] * 100)

        # Default model is the best
        # We use the precision, recall and F1
        # between 0 and 100: 100 is the best
        bertscore_score = bertscore.compute(predictions=simplification, references=original, lang="en", device=device)
        bertscore_precision_scores.append(bertscore_score["precision"][0] * 100)
        bertscore_recall_scores.append(bertscore_score["recall"][0] * 100)
        bertscore_f1_scores.append(bertscore_score["f1"][0] * 100)

        # between 0 and 100: 100 is the best
        questeval_scores.append(
            questeval.corpus_questeval(sources=original, hypothesis=simplification)["ex_level_scores"][0] * 100
        )

        # between 0 and 100: 100 is the best
        lens_scores.append(
            lens.score(
                complex=original,
                simplified=simplification,
                references=[original],
                batch_size=1,
                gpus=1,
            )[0]
        )

        # between 0 and 100: 100 is the best
        coverage_scores.append(coverage_kis.score(bodies=original, decodeds=simplification)["scores"][0] * 100)

        # between o and 100: 100 is the best
        sentence_transformer_scores.append(
            sentence_transformer.compute(prediction=simplification[0], reference=original[0]).detach().cpu().item()
            * 100
        )

        ter_scores.append(100 - ter.compute(predictions=simplification, references=[original])["score"])

        rouge_score = rouge.compute(predictions=simplification, references=original)
        rouge_1_scores.append(rouge_score["rouge1"] * 100)
        rouge_2_scores.append(rouge_score["rouge2"] * 100)
        rouge_l_scores.append(rouge_score["rougeL"] * 100)

    # Compilation of the scores per metrics
    for metric_name, metric_score in zip(
        [
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
        ],
        [
            sari_scores,
            bleu_scores,
            bleu_sari_am_scores,
            bleu_sari_gm_scores,
            bleurt_scores,
            ibleu_scores,
            fkbleu_scores,
            meteor_scores,
            bertscore_precision_scores,
            bertscore_recall_scores,
            bertscore_f1_scores,
            questeval_scores,
            lens_scores,
            coverage_scores,
            sentence_transformer_scores,
            ter_scores,
            rouge_1_scores,
            rouge_2_scores,
            rouge_l_scores,
        ],
    ):
        counts_1 = [round(s) < 1 for s in metric_score]
        ratio_1 = sum(counts_1) / len(counts_1) * 100
        counts_5 = [round(s) < 5 for s in metric_score]
        ratio_5 = sum(counts_5) / len(counts_5) * 100
        counts_equals = [round(s) == 0 for s in metric_score]
        ratio_equals = sum(counts_equals) / len(counts_equals) * 100

        logger.log({f"test/irrelevant_sentence_{metric_name}_ratio_equals": ratio_equals})
        logger.log({f"test/irrelevant_sentence_{metric_name}_ratio_1": ratio_1})
        logger.log({f"test/irrelevant_sentence_{metric_name}_ratio_5": ratio_5})

        rmse_score = mean_squared_error(metric_score, ratings, squared=False)
        logger.log({f"test/irrelevant_sentence{metric_name}_rmse": rmse_score})

        logger.log({f"test/irrelevant_sentence{metric_name}_mean": mean(metric_score)})
        logger.log({f"test/irrelevant_sentence{metric_name}_st_dev": stdev(metric_score)})
