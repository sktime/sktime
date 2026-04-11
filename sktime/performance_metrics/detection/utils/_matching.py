"""Matching utilities for detection metrics."""


def _count_windowed_matches(y_true, y_pred, margin):
    """Count one-to-one matches between true and predicted event ilocs.

    Parameters
    ----------
    y_true : array-like
        Ground truth event ilocs.
    y_pred : array-like
        Predicted event ilocs.
    margin : int
        Maximum absolute iloc distance for a predicted event to match a true event.

    Returns
    -------
    int
        Maximum number of greedy one-to-one matches under the margin criterion.
    """
    if margin < 0:
        raise ValueError("margin must be a non-negative integer")

    true_ilocs = sorted(y_true)
    pred_ilocs = sorted(y_pred)

    true_idx = 0
    pred_idx = 0
    matches = 0

    while true_idx < len(true_ilocs) and pred_idx < len(pred_ilocs):
        true_iloc = true_ilocs[true_idx]
        pred_iloc = pred_ilocs[pred_idx]

        if abs(pred_iloc - true_iloc) <= margin:
            matches += 1
            true_idx += 1
            pred_idx += 1
        elif pred_iloc < true_iloc - margin:
            pred_idx += 1
        else:
            true_idx += 1

    return matches
