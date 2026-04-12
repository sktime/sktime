"""Early detection score for time series event detection tasks."""

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class EarlyDetectionScore(BaseDetectionMetric):
    r"""Early Detection Score for the first event in time series event detection.

    This metric rewards detecting the first true event early or on time.
    Traditional metrics like accuracy or F1-score do not account for *when*
    a detection occurs. It is particularly useful in real-world applications
    (e.g., sensor monitoring in agriculture machinery) where early alerts
    provide high value.

    The score is computed **only on the first event** (smallest `ilocs` value):

    - Find the smallest event index in ``y_true`` and ``y_pred``.
    - ``delay = pred_first - true_first``
    - If ``delay <= 0`` (early or on time): score = 1.0
    - If delayed: score = 1 / (1 + delay)
    - If no events in ``y_true`` or no detection in ``y_pred``: score = 0.0

    Parameters
    ----------
    None
        Kept parameter-free for simplicity.
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",  # event positions via 'ilocs' column
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,  # higher score is better
    }

    def __init__(self):
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute Early Detection Score on the first events.

        Parameters
        ----------
        y_true, y_pred : pd.DataFrame
            Must contain column 'ilocs' with integer event positions
            (coerced by the base class).
        X : pd.DataFrame, optional (default=None)
            Not used.

        Returns
        -------
        float
            Early detection score in [0, 1].
        """
        # Handle edge cases: no true events or no predictions
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0

        # First (earliest) event
        true_first = min(y_true["ilocs"].values)
        pred_first = min(y_pred["ilocs"].values)

        delay = pred_first - true_first

        if delay <= 0:
            return 1.0
        else:
            return 1.0 / (1.0 + delay)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the metric."""
        return [{}]  # no parameters
