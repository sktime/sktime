import numpy as np


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
