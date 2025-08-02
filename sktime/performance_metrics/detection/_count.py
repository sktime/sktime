"""Count of detection, possibly in excess or deviation of a target count."""

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class DetectionCount(BaseDetectionMetric):
    r"""Count of detection, possibly in excess or deviation of a target count.

    Parameters
    ----------
    target : int, default=0
        Target number of detections.
        If 0, the count is the absolute number of detections.
        If positive, the count reported is, by default, the absolute
        difference between the number of detections and the target.
        If the ``excess_only`` parameter is set to ``True``, the count
        is the number of detections in excess of the target if larger,
        otherwise zero.
    excess_only : bool, default=False
        If False, the count is the absolute difference between the ``target``
        and the number of detections.
        If True, the count is the number of detections in excess of the target
        if larger, otherwise zero.
    """

    _tags = {
        "scitype:y": "points",  # or segments
        "requires_X": False,
        "requires_y_true": False,  # this is an unsupervised metric
        "lower_is_better": True,
    }

    def __init__(self, target=0, excess_only=False):
        self.target = target
        self.excess_only = excess_only

        super().__init__()

    def _evaluate(self, y_true, y_pred, X):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate

        Parameters
        ----------
        y_true : time series in ``sktime`` compatible data container format.
            Ground truth (correct) event locations, in ``X``.
            Should be ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D),
            of ``Series`` scitype = individual time series.

            For further details on data format, see glossary on :term:`mtype`.

        y_pred : time series in ``sktime`` compatible data container format
            Detected events to evaluate against ground truth.
            Must be of same format as ``y_true``, same indices and columns if indexed.

        X : optional, pd.DataFrame, pd.Series or np.ndarray
            Time series that is being labelled.
            If not provided, assumes ``RangeIndex`` for ``X``, and that
            values in ``X`` do not matter.

        Returns
        -------
        loss : float
            Calculated metric.
        """
        num_detections = len(y_pred)

        score = num_detections - self.target

        if self.excess_only:
            score = max(score, 0)

        score = abs(score)

        return score

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
        param1 = {"target": 1, "excess_only": True}
        param2 = {"target": 42}

        return [param0, param1, param2]
