"""Dummy anomaly detector which detects anomalies after steps."""

import pandas as pd

from sktime.detection.base import BaseDetector


class DummyRegularAnomalies(BaseDetector):
    """Dummy change point detector which detects a change point every x steps.

    Naive method that can serve as benchmarking pipeline or API test.

    Detects a change point every ``step_size`` location indices.
    The first change point is detected at location index ``step_size``,
    the second at ``2 * step_size``, and so on.

    Parameters
    ----------
    step_size : int, default=2
        The step size at which change points are detected.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.dummy import DummyRegularAnomalies
    >>> y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> d = DummyRegularAnomalies(step_size=3)
    >>> yt = d.fit_transform(y)
    """

    _tags = {
        "authors": ["fkiraly"],
        "capability:multivariate": True,
        "capability:missing_values": True,
        "fit_is_empty": False,
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, step_size=2):
        self.step_size = step_size
        super().__init__()

    def _fit(self, X, y=None):
        """Fit to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to time series.
        y : pd.Series, optional
            Ground truth labels for training if detector is supervised.

        Returns
        -------
        self :
            Reference to self.
        """
        X_index = X.index
        if isinstance(X_index, pd.DatetimeIndex):
            X_index = pd.PeriodIndex(X_index)
        self.first_index_ = X_index[0]
        return self

    def _predict(self, X):
        """Create labels on test/deployment data.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              the values are integer indices of the changepoints/anomalies.
            * If ``task`` is "segmentation", the values are ``pd.Interval`` objects.
        """
        step_size = self.step_size

        X_index = X.index
        if isinstance(X_index, pd.DatetimeIndex):
            X_index = pd.PeriodIndex(X_index)

        first_index = self.first_index_

        offset = X_index - first_index

        # this handles PeriodIndex, by converting to integer
        if offset.dtype == "object":
            offset = [o.n for o in offset]
            offset = pd.Index(offset)

        change_point_indicator = offset % step_size == step_size - 1

        X_range = pd.RangeIndex(len(X))
        X_ix_cp = X_range[change_point_indicator]
        change_points = pd.Series(X_ix_cp, dtype="int64")
        return change_points

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params0 = {}
        params1 = {"step_size": 3}
        params2 = {"step_size": 100}

        return [params0, params1, params2]
