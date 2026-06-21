"""Matching utilities for detection metrics."""


def _count_windowed_matches(y_true, y_pred, margin_backward=0, margin_forward=None):
    """Count one-to-one matches between true and predicted event ilocs.

    Parameters
    ----------
    y_true : array-like
        Ground truth event ilocs.
    y_pred : array-like
        Predicted event ilocs.
    margin_backward : int or timedelta-like, optional (default=0)
        Maximum backward distance for a predicted event to match a true event.
    margin_forward : int or timedelta-like, optional (default=None)
        Maximum forward distance for a predicted event to match a true event.
        If None, uses ``margin_backward``.

    Returns
    -------
    int
        Maximum number of greedy one-to-one matches under the margin criterion.
    """
    if margin_forward is None:
        margin_forward = margin_backward

    zero_backward = margin_backward - margin_backward
    zero_forward = margin_forward - margin_forward

    if margin_backward < zero_backward or margin_forward < zero_forward:
        raise ValueError("window margins must be non-negative")

    true_ilocs = sorted(y_true)
    pred_ilocs = sorted(y_pred)

    true_idx = 0
    pred_idx = 0
    matches = 0

    while true_idx < len(true_ilocs) and pred_idx < len(pred_ilocs):
        true_iloc = true_ilocs[true_idx]
        pred_iloc = pred_ilocs[pred_idx]

        if pred_iloc < true_iloc - margin_backward:
            pred_idx += 1
        elif pred_iloc <= true_iloc + margin_forward:
            matches += 1
            true_idx += 1
            pred_idx += 1
        else:
            true_idx += 1

    return matches
