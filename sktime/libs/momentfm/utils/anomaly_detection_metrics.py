"""Anomaly detection metrics."""

import numpy as np


def f1_score(predict, actual):
    """Calculate f1 score."""
    TP = np.sum(predict * actual)
    # TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1


def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """Calculate Adjusted predict labels.

    Calculate adjusted predict labels using given `score`,
    `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the
            threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore
        `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def adjbestf1(y_true: np.array, y_scores: np.array, n_splits: int = 100):
    """Calculate adjusted best f1."""
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_splits)
    adjusted_f1 = np.zeros(thresholds.shape)

    for i, threshold in enumerate(thresholds):
        y_pred = y_scores >= threshold
        y_pred = adjust_predicts(
            score=y_scores,
            label=(y_true > 0),
            pred=y_pred,
            threshold=None,
            calc_latency=False,
        )
        adjusted_f1[i] = f1_score(y_pred, y_true)

    best_adjusted_f1 = np.max(adjusted_f1)
    return best_adjusted_f1
