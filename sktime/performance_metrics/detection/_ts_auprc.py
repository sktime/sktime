import numpy as np
from sklearn.metrics import auc

from sktime.performance_metrics.detection._base import BaseDetectionMetric


def _constant_bias_fn(inputs: np.ndarray):
    """Compute the overlap size for a constant bias function that assigns the.

    same weight to all positions.

    This function computes the average of the input values.
    :inputs: A 1-D numpy array containing the predictions inside a
                  ground-truth window.
    :return: The overlap, which is the average of the input values.
    """
    return np.sum(inputs) / inputs.shape[0]


def _inverse_proportional_cardinality_fn(cardinality: int, gt_length: int):
    """Cardinality that assigns an inversely proportional weight to predictions.

    within a single ground-truth window.

    This is the default cardinality function recommended in [Tatbul2018]
    [Tatbul2018] N. Tatbul, T.J. Lee, S. Zdonik, M. Alam, J. Gottschlich.
    Precision and recall for time series.
    Advances in neural information processing systems.

    [Wagner2023]D. Wagner, T. Michels, F.C.F. Schulz, A. Nair, M. Rudolph, and M. Kloft.
    TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection.
    Transactions on Machine Learning Research (TMLR), (to appear) 2023.

    :cardinality: No. of predicted windows that overlap ground-truth window in question.
    :gt_length: Length of the ground-truth window (unused).
    :return: The cardinality factor 1/cardinality, with a minimum value of 1.
    """
    return 1 / max(1, cardinality)


def _improved_cardinality_fn(cardinality: int, gt_length: int):
    """Compute recall-consistent cardinality factor from [Wagner2023]_.

    ((gt_length - 1) / gt_length) ** (cardinality - 1)
    TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection, TMLR, 2023.
    :cardinality: Count of predicted windows overlapping the ground-truth window.
    :gt_length: Length of the ground-truth window.
    :return: Cardinality factor.
    """
    return ((gt_length - 1) / gt_length) ** (cardinality - 1)


def _compute_window_indices(binary_labels: np.ndarray):
    """Get anomaly window indices from binary labels.

    :binary_labels: 1-D numpy array with 1 for anomalies, 0 otherwise.
    :return: List of (start, end) tuples, where start is the window's start index and
            end is the first index after its end.
    """
    # Compute the differences between consecutive elements
    differences = np.diff(binary_labels, prepend=0)
    # Find the indices where the differences are non-zero (start and end of windows)
    indices = np.nonzero(differences)[0]
    # If the number of indices is odd, append the last index
    if len(indices) % 2 != 0:
        indices = np.append(indices, binary_labels.size)
    # Pair the start and end indices
    window_indices = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]

    return window_indices


def _compute_overlap(
    preds: np.ndarray,
    pred_indices,
    gt_indices,
    alpha: float,
    bias_fn,
    cardinality_fn,
    use_window_weight,
) -> float:
    n_gt_windows = len(gt_indices)
    n_pred_windows = len(pred_indices)
    total_score = 0.0
    total_gt_points = 0

    i = j = 0
    while i < n_gt_windows and j < n_pred_windows:
        gt_start, gt_end = gt_indices[i]
        window_length = gt_end - gt_start
        total_gt_points += window_length
        i += 1

        cardinality = 0
        while j < n_pred_windows and pred_indices[j][1] <= gt_start:
            j += 1
        while j < n_pred_windows and pred_indices[j][0] < gt_end:
            j += 1
            cardinality += 1

        if cardinality == 0:
            # cardinality == 0 means no overlap at all, hence no contribution
            continue

        # The last predicted window that overlaps our current window could
        # also overlap the next window.
        # Therefore, we must consider it again in the next loop iteration.
        j -= 1

        cardinality_multiplier = cardinality_fn(cardinality, window_length)

        prediction_inside_ground_truth = preds[gt_start:gt_end]
        # We calculate omega directly in the bias function,
        # because this can greatly improve running time

        # for the constant bias, for example.
        omega = bias_fn(prediction_inside_ground_truth)

        # Either weight evenly across all windows or based on window length
        weight = window_length if use_window_weight else 1

        # Existence reward (if cardinality > 0 then this is certainly 1)
        total_score += alpha * weight
        # Overlap reward
        total_score += (1 - alpha) * cardinality_multiplier * omega * weight

    denom = total_gt_points if use_window_weight else n_gt_windows

    return total_score / denom


def _ts_precision_and_recall(
    anomalies: np.ndarray,
    predictions: np.ndarray,
    alpha: float = 0,
    recall_bias_fn=_constant_bias_fn,
    recall_cardinality_fn=_inverse_proportional_cardinality_fn,
    precision_bias_fn=None,
    precision_cardinality_fn=None,
    anomaly_ranges=None,
    prediction_ranges=None,
    weighted_precision=False,
):
    """Compute time-series precision and recall as defined in [Tatbul2018]_.

    :anomalies: Binary 1-D numpy array of true labels.
    :predictions: Binary 1-D numpy array of predicted labels.
    :alpha: Weight for recall's existence term.
    :recall_bias_fn: Function for recall bias term per ground-truth window.
    :recall_cardinality_fn: Function for recall cardinality factor.
    :precision_bias_fn: Function for precision bias term;
    :precision_cardinality_fn: Function for precision cardinality factor;
                                     defaults to recall_cardinality_fn.
    :weighted_precision: Weight precision by window length if True.
    :anomaly_ranges: List of (start, end) tuples for true anomaly windows;
    :prediction_ranges: List of (start, end) tuples for predicted windows;
    :return: Tuple of time-series precision and recall.
    """
    has_anomalies = np.any(anomalies > 0)
    has_predictions = np.any(predictions > 0)

    # Catch special cases which would cause a division by zero
    if not has_predictions and not has_anomalies:
        # classifier is perfect, so it makes sense to set precision and recall to 1
        return 1, 1
    elif not has_predictions or not has_anomalies:
        return 0, 0

    # Set precision functions to the same as recall functions if they are not given
    if precision_bias_fn is None:
        precision_bias_fn = recall_bias_fn
    if precision_cardinality_fn is None:
        precision_cardinality_fn = recall_cardinality_fn

    if anomaly_ranges is None:
        anomaly_ranges = _compute_window_indices(anomalies)
    if prediction_ranges is None:
        prediction_ranges = _compute_window_indices(predictions)

    recall = _compute_overlap(
        predictions,
        prediction_ranges,
        anomaly_ranges,
        alpha,
        recall_bias_fn,
        recall_cardinality_fn,
    )
    precision = _compute_overlap(
        anomalies,
        anomaly_ranges,
        prediction_ranges,
        0,
        precision_bias_fn,
        precision_cardinality_fn,
        use_window_weight=weighted_precision,
    )

    return precision, recall


def _ts_auprc(y_true, scores, integration="trapezoid", weighted_precision=True):
    thresholds = np.unique(scores)
    precision = np.empty(len(thresholds) + 1)
    recall = np.empty(len(thresholds) + 1)
    predictions = np.empty_like(scores, dtype=int)

    precision[-1] = 1
    recall[-1] = 0
    label_ranges = _compute_window_indices(y_true)
    for i, t in enumerate(thresholds):
        predictions = scores >= t
        prec, rec = _ts_precision_and_recall(
            y_true,
            predictions,
            alpha=0,
            recall_cardinality_fn=_improved_cardinality_fn,
            anomaly_ranges=label_ranges,
            weighted_precision=weighted_precision,
        )
        precision[i] = prec
        recall[i] = rec
    if integration == "riemann":
        area = -np.sum(np.diff(recall) * precision[:-1])
    else:
        area = auc(recall, precision)

    return area, {}


class TimeSeriesAUPRC(BaseDetectionMetric):
    """Compute the area under the precision-recall curve for time series."""

    def __init__(self, integration="trapezoid", weighted_precision=True):
        self.integration = integration
        self.weighted_precision = weighted_precision

        super().__init__()

    def _evaluate(self, y_true, scores, X=None):
        self._integration = self.integration
        self._weighted_precision = self.weighted_precision
        return _ts_auprc(
            y_true,
            scores,
            integration=self._integration,
            weighted_precision=self._weighted_precision,
        )
