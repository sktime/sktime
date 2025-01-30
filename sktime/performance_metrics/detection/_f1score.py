from sktime.performance_metrics.detection._base import BaseDetectionMetric


class WindowedF1Score(BaseDetectionMetric):
    """F1-score for event detection, using a margin-based match criterion.

    This score computes an iloc-based F1-score with margin of errog

    A true event is considered a match if there is a predicted
    event within a margin of error, as specified by the ``margin`` parameter.

    The margin is applied in ``iloc`` units, i.e., the absolute difference between
    the true and predicted iloc must be less or equal than the margin to be considered
    a match.

    The default value of 0 results in only exact matches being counted.

    Parameters
    ----------
    margin : int, optional (default=0)
        Margin of error to consider a detected event as matched.
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",  # expects each row to represent one breakpoint
        "requires_X": False,  # not using X by default
        "requires_y_true": True,  # supervised metric
        "lower_is_better": False,  # higher F1 is better
    }

    def __init__(self, margin=0):
        self.margin = margin
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute F1 score under the margin-based breakpoint matching logic.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth breakpoints in "points" format. Must have column 'ilocs'.
        y_pred : pd.DataFrame
            Predicted breakpoints in "points" format. Must have column 'ilocs'.
        X : pd.DataFrame, optional (default=None)
            Unused here, but part of the signature.

        Returns
        -------
        float
            F1 score, i.e., 2 * precision * recall / (precision + recall).
        """
        # Convert breakpoints to sorted Python lists
        # We assume the DataFrame has a column 'ilocs' with integer or float index positions  # noqa: E501
        gt = sorted(y_true["ilocs"].values)
        pred = sorted(y_pred["ilocs"].values)

        # Handle edge cases
        if len(gt) == 0 and len(pred) == 0:
            return 1.0  # No breakpoints to detect, so consider it perfect by convention
        if len(gt) == 0:
            return 0.0  # ground truth is empty but predictions exist => precision = 0 => F1=0  # noqa: E501
        if len(pred) == 0:
            return 0.0  # predictions are empty but ground truth exists => recall = 0 => F1=0  # noqa: E501

        # Count how many ground-truth breakpoints are matched
        matched_count = 0
        pred_index = 0

        margin = self.margin

        for true_bkpt in gt:
            # Advance pred_index while predicted < (true - margin)
            while pred_index < len(pred) and pred[pred_index] <= true_bkpt - margin:
                pred_index += 1

            # If current predicted is within margin, count it as matched
            if pred_index < len(pred) and abs(pred[pred_index] - true_bkpt) <= margin:
                matched_count += 1
                # optional: pred_index += 1 to avoid double-matching the same prediction

        # Compute precision and recall
        precision = matched_count / len(pred)
        recall = matched_count / len(gt)

        # Compute F1
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        param0 = {}
        param1 = {"margin": 1}
        param2 = {"margin": 42424242424242}

        return [param0, param1, param2]
