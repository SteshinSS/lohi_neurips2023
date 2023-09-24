import numpy as np
import pandas as pd
import os
from deepchem.metrics import bedroc_score
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score, mean_absolute_error
from scipy.special import expit as sigmoid
from scipy.stats import spearmanr


def top_recall(y: np.ndarray, log_proba: np.ndarray, percentile: float):
    percentile = np.percentile(log_proba, q=percentile)
    top_idx = log_proba > percentile
    total_top_elements = top_idx.sum()
    if total_top_elements == 0:
        return 0
    else:
        return y[top_idx].sum() / y.sum()


def simulate_experiment(y: np.ndarray, log_proba: np.ndarray, n: int):
    best_idx = np.argsort(log_proba)[-n:]
    total_hits = y[best_idx].sum()
    return total_hits


def get_hi_metrics(data: pd.DataFrame, y_pred: np.ndarray):
    """Calculates Hi classification metrics

    Args:
        y_pred (np.ndarray): 1d vector of log probabilities of positive class

    Returns:
        dict: with metrics
    """
    y = data['value'].astype(float)
    y_prob = sigmoid(y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    prc_auc = average_precision_score(y, y_prob)

    two_class_prob = np.stack([1.0 - y_prob, y_prob]).transpose()
    bedroc = bedroc_score(y, two_class_prob, alpha=70.0)

    metrics = {
        "roc_auc": float(roc_auc),
        "bedroc": float(bedroc),
        "prc_auc": float(prc_auc),
    }
    return metrics


def get_lo_metrics(data, y_pred):
    data = data.copy()
    data['preds'] = y_pred

    r2_scores = []
    spearman_scores = []
    maes = []
    for cluster_idx in data['cluster'].unique():
        cluster = data[data['cluster'] == cluster_idx]
        r2 = r2_score(cluster['value'], cluster['preds'])
        r2_scores.append(r2)

        spearman, _ = spearmanr(cluster['value'], cluster['preds'])
        if np.isnan(spearman):
            spearman = 0.0
        spearman_scores.append(spearman)

        mae = mean_absolute_error(cluster['value'], cluster['preds'])
        maes.append(mae)

    r2_scores = np.array(r2_scores)
    spearman_scores = np.array(spearman_scores)
    maes = np.array(maes)

    metrics = {
        'r2': r2_scores.mean(),
        'spearman': spearman_scores.mean(),
        'mae': maes.mean()
    }
    return metrics


def get_list_of_methods(predictions_path):
    folders = []
    for folder in os.listdir(predictions_path):
        if os.path.isdir(os.path.join(predictions_path, folder)):
            folders.append(folder)
    return folders

def summarize_metrics(metrics_per_split):
    metric_names = metrics_per_split[0].keys()

    result_mean = {}
    result_std = {}

    for metric in metric_names:
        values = []
        for split in metrics_per_split:
            values.append(split[metric])
        values = np.array(values)
        result_mean[metric] = values.mean()
        result_std[metric] = values.std()
    return result_mean, result_std


def get_summary_metrics(predictions_path, methods, get_metrics_func): 
    train_means = []
    train_stds = []
    test_means = []
    test_stds = []

    for method in methods:
        results_path = predictions_path + method
        train_metrics = []
        test_metrics = []

        for i in [1, 2, 3]:
            train = pd.read_csv(results_path + f'/train_{i}.csv')
            train_metrics.append(get_metrics_func(train, train['preds']))

            test = pd.read_csv(results_path + f'/test_{i}.csv')
            test_metrics.append(get_metrics_func(test, test['preds']))
        
        train_mean, train_std = summarize_metrics(train_metrics)
        train_means.append(train_mean)
        train_stds.append(train_std)

        test_mean, test_std = summarize_metrics(test_metrics)
        test_means.append(test_mean)
        test_stds.append(test_std)
    return train_means, train_stds, test_means, test_stds


def compile_summary_table(train_means, train_stds, test_means, test_stds, methods):
    result = {
        'method': methods,
    }

    keys = train_means[0].keys()
    for key in keys:
        # train column
        column = []
        for i in range(len(train_means)):
            mean = round(train_means[i][key], 3)
            std = round(train_stds[i][key], 3)
            column.append(str(mean) + '±' + str(std))
        result[key + '_train'] = column

        # test column
        column = []
        for i in range(len(test_means)):
            mean = round(test_means[i][key], 3)
            std = round(test_stds[i][key], 3)
            column.append(str(mean) + '±' + str(std))
        result[key + '_test'] = column      

    return pd.DataFrame(result)  