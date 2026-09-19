"""Range-based Precision and Recall for advance event detection."""

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class RangeBasedPrecision(BaseDetectionMetric):
    r"""Range-based Precision for event detection with positional bias.

    Extends classical Precision to range-based events with a customizable
    positional bias function. With ``bias="front"``, detections earlier
    in the true event range receive higher scores, directly rewarding
    advance detection.

    Based on Tatbul et al., NeurIPS 2018.
    https://arxiv.org/abs/1803.03639

    Parameters
    ----------
    bias : str, default="flat"
        Positional bias function. One of "flat", "front", "back".
        "flat"  - all positions within a range are equally weighted.
        "front" - earlier positions in a range receive higher weight,
                  rewarding advance/early detection.
        "back"  - later positions in a range receive higher weight.
    alpha : float, default=0.0
        Weight for existence reward (0 to 1). Default 0 means
        the score is purely based on overlap.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection._range_based_precision_recall import (
    ...     RangeBasedPrecision,
    ... )
    >>> y_true = pd.DataFrame({"ilocs": [10, 11, 12, 13, 14]})
    >>> y_pred = pd.DataFrame({"ilocs": [10, 11]})
    >>> metric = RangeBasedPrecision(bias="front")
    >>> metric(y_true, y_pred)  # doctest: +SKIP
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,  # higher precision is better
    }

    def __init__(self, bias="flat", alpha=0.0):
        self.bias = bias
        self.alpha = alpha
        super().__init__()

    def _positional_bias(self, position, length):
        """Compute positional weight for a detection within a range.

        Parameters
        ----------
        position : int
            Position within the range (0-indexed from start).
        length : int
            Total length of the range.

        Returns
        -------
        float
            Weight for this position.
        """
        if length <= 1:
            return 1.0

        if self.bias == "flat":
            return 1.0
        elif self.bias == "front":
            # linearly decreasing: earlier positions get higher weight
            return float(length - position) / length
        elif self.bias == "back":
            # linearly increasing: later positions get higher weight
            return float(position + 1) / length
        else:
            raise ValueError(
                f"Unknown bias '{self.bias}'. Choose from 'flat', 'front', 'back'."
            )

    def _overlap_score(self, pred_iloc, true_ilocs_set, true_start, true_end):
        """Compute overlap score for a single predicted point.

        Parameters
        ----------
        pred_iloc : int
            The predicted event position.
        true_ilocs_set : set
            Set of all true event positions.
        true_start : int
            Start of the true event range.
        true_end : int
            End of the true event range (inclusive).

        Returns
        -------
        float
            Overlap score between 0 and 1.
        """
        if pred_iloc not in true_ilocs_set:
            return 0.0

        range_length = true_end - true_start + 1
        position = pred_iloc - true_start
        weight = self._positional_bias(position, range_length)

        # normalize weights across the range
        total_weight = sum(
            self._positional_bias(p, range_length) for p in range(range_length)
        )
        if total_weight == 0:
            return 0.0

        return weight / total_weight

    def _evaluate(self, y_true, y_pred, X=None):
        """Evaluate Range-based Precision on given inputs.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth events in "points" format with column 'ilocs'.
        y_pred : pd.DataFrame
            Predicted events in "points" format with column 'ilocs'.
        X : pd.DataFrame, optional
            Unused, part of signature.

        Returns
        -------
        float
            Range-based Precision score between 0.0 and 1.0.
        """
        true_ilocs = sorted(y_true["ilocs"].values.tolist())
        pred_ilocs = sorted(y_pred["ilocs"].values.tolist())

        if len(pred_ilocs) == 0 and len(true_ilocs) == 0:
            return 1.0
        if len(pred_ilocs) == 0:
            return 0.0
        if len(true_ilocs) == 0:
            return 0.0

        true_ilocs_set = set(true_ilocs)
        true_start = min(true_ilocs)
        true_end = max(true_ilocs)

        total_score = 0.0
        for pred in pred_ilocs:
            # existence reward: does any true event exist?
            existence = 1.0 if len(true_ilocs) > 0 else 0.0
            # overlap reward
            overlap = self._overlap_score(pred, true_ilocs_set, true_start, true_end)
            total_score += self.alpha * existence + (1 - self.alpha) * overlap

        return total_score / len(pred_ilocs)

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
        param1 = {}
        param2 = {"bias": "front"}
        param3 = {"bias": "back", "alpha": 0.1}
        return [param1, param2, param3]


class RangeBasedRecall(BaseDetectionMetric):
    r"""Range-based Recall for event detection with positional bias.

    Extends classical Recall to range-based events with a customizable
    positional bias function. With ``bias="front"``, detections earlier
    in the true event range contribute more to recall, directly rewarding
    advance/early detection.

    Based on Tatbul et al., NeurIPS 2018.
    https://arxiv.org/abs/1803.03639

    Parameters
    ----------
    bias : str, default="flat"
        Positional bias function. One of "flat", "front", "back".
        "flat"  - all positions within a range are equally weighted.
        "front" - earlier positions in a range receive higher weight,
                  rewarding advance/early detection.
        "back"  - later positions in a range receive higher weight.
    alpha : float, default=0.0
        Weight for existence reward (0 to 1).

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection._range_based_precision_recall import (
    ...     RangeBasedRecall,
    ... )
    >>> y_true = pd.DataFrame({"ilocs": [10, 11, 12, 13, 14]})
    >>> y_pred = pd.DataFrame({"ilocs": [10, 11]})
    >>> metric = RangeBasedRecall(bias="front")
    >>> metric(y_true, y_pred)  # doctest: +SKIP
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,  # higher recall is better
    }

    def __init__(self, bias="flat", alpha=0.0):
        self.bias = bias
        self.alpha = alpha
        super().__init__()

    def _positional_bias(self, position, length):
        """Compute positional weight for a detection within a range."""
        if length <= 1:
            return 1.0

        if self.bias == "flat":
            return 1.0
        elif self.bias == "front":
            return float(length - position) / length
        elif self.bias == "back":
            return float(position + 1) / length
        else:
            raise ValueError(
                f"Unknown bias '{self.bias}'. Choose from 'flat', 'front', 'back'."
            )

    def _evaluate(self, y_true, y_pred, X=None):
        """Evaluate Range-based Recall on given inputs.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth events in "points" format with column 'ilocs'.
        y_pred : pd.DataFrame
            Predicted events in "points" format with column 'ilocs'.
        X : pd.DataFrame, optional
            Unused, part of signature.

        Returns
        -------
        float
            Range-based Recall score between 0.0 and 1.0.
        """
        true_ilocs = sorted(y_true["ilocs"].values.tolist())
        pred_ilocs = sorted(y_pred["ilocs"].values.tolist())

        if len(true_ilocs) == 0 and len(pred_ilocs) == 0:
            return 1.0
        if len(true_ilocs) == 0:
            return 0.0
        if len(pred_ilocs) == 0:
            return 0.0

        pred_ilocs_set = set(pred_ilocs)
        true_start = min(true_ilocs)
        true_end = max(true_ilocs)
        range_length = true_end - true_start + 1

        # existence reward
        existence = 1.0 if len(pred_ilocs) > 0 else 0.0

        # overlap reward: sum weighted contributions of true events that are detected
        total_weight = sum(
            self._positional_bias(t - true_start, range_length) for t in true_ilocs
        )

        if total_weight == 0:
            return 0.0

        overlap_score = 0.0
        for true in true_ilocs:
            if true in pred_ilocs_set:
                position = true - true_start
                weight = self._positional_bias(position, range_length)
                overlap_score += weight / total_weight

        recall = self.alpha * existence + (1 - self.alpha) * overlap_score
        return recall

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
        param1 = {}
        param2 = {"bias": "front"}
        param3 = {"bias": "back", "alpha": 0.1}
        return [param1, param2, param3]