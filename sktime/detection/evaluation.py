"""Evaluation utilities for detection models."""

__all__ = ["evaluate_detection"]


def evaluate_detection(true_events, predicted_events, tolerance=5):
    """Evaluate detection performance using precision and recall.

    Parameters
    ----------
    true_events : list of int
        Ground truth event indices.
    predicted_events : list of int
        Predicted event indices.
    tolerance : int, optional (default=5)
        Allowed distance for matching events.

    Returns
    -------
    precision : float
    recall : float
    """
    matched = 0
    used_preds = set()

    for t in true_events:
        for i, p in enumerate(predicted_events):
            if i in used_preds:
                continue
            if abs(t - p) <= tolerance:
                matched += 1
                used_preds.add(i)
                break

    precision = matched / len(predicted_events) if predicted_events else 0.0
    recall = matched / len(true_events) if true_events else 0.0

    return precision, recall