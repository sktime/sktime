import numpy as np

from sktime.performance_metrics.detection.utils._window import (
    _compute_overlap,
    _compute_window_indices,
)


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
        use_window_weight=weighted_precision,
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
