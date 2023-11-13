from evaluate import load
from sklearn.metrics import mean_squared_error

r2_metric = load("r_squared")
pearsonr_metric = load("pearsonr")


def compute_metrics(eval_pred):
    """
    Function to compute various metric during training.
    """
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    r_squared = r2_metric.compute(predictions=predictions, references=labels)
    pearson_corr = pearsonr_metric.compute(predictions=predictions, references=labels, return_pvalue=True)
    mean_score = predictions.mean()
    st_dev_score = predictions.std()
    return {
        "rmse": rmse,
        "R2": r_squared,
        "pearson_corr": pearson_corr["pearsonr"],
        "pearson_pvalue": pearson_corr["p-value"],
        "mean_score": mean_score,
        "st_dev_score": st_dev_score,
    }


def eval_compute_metrics_identical(eval_pred):
    """
    Function to compute various metric during evaluation for identical sentences.
    """
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    r_squared = r2_metric.compute(predictions=predictions, references=labels)
    mean_score = predictions.mean()
    st_dev_score = predictions.std()

    # This is only for the hold out test
    counts_95 = [s.round() > 95 for s in predictions]
    ratio_95 = sum(counts_95) / len(counts_95) * 100
    counts_99 = [s.round() > 99 for s in predictions]
    ratio_99 = sum(counts_99) / len(counts_99) * 100
    counts_equals = [s.round() == 100 for s in predictions]
    ratio_equals = sum(counts_equals) / len(counts_equals) * 100
    return {
        "rmse": rmse,
        "R2": r_squared,
        "mean_score": mean_score,
        "st_dev_score": st_dev_score,
        "ratio_equals": ratio_equals,
        "ratio_95": ratio_95,
        "ratio_99": ratio_99,
    }


def eval_compute_metrics_unrelated(eval_pred):
    """
    Function to compute various metric during evaluation for unrelated sentences.
    """
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    r_squared = r2_metric.compute(predictions=predictions, references=labels)
    mean_score = predictions.mean()
    st_dev_score = predictions.std()

    # This is only for the hold out test
    counts_1 = [s.round() < 1 for s in predictions]
    ratio_1 = sum(counts_1) / len(counts_1) * 100
    counts_5 = [s.round() < 5 for s in predictions]
    ratio_5 = sum(counts_5) / len(counts_5) * 100
    counts_equals = [s.round() == 0 for s in predictions]
    ratio_equals = sum(counts_equals) / len(counts_equals) * 100
    return {
        "rmse": rmse,
        "R2": r_squared,
        "mean_score": mean_score,
        "st_dev_score": st_dev_score,
        "ratio_equals": ratio_equals,
        "ratio_1": ratio_1,
        "ratio_5": ratio_5,
    }
