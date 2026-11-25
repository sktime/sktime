"""Hampel filter for univariate anomaly detection."""

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector

__author__ = ["SpitFire19"]


class HampelDetector(BaseDetector):
    """
    Anomaly detector based on Hampel filter described in [1].

    Parameters
    ----------
    window_length : int, optional (default=10)
        The size of the sliding window (number of observations).
    n_sigma : float, optional (default=3.0)
        The number of standard deviations to use for the outlier threshold.
    k : float, optional (default=1.4826)
        The consistency constant which depends on the underlying distribution.
        By default, we choose k=1.4826 - the value for Gaussian distribution.
    causal : bool, default=False
        If False, use centered windows (offline mode).
        If True, use trailing windows that depend only on
        the past (causally correct online mode).
        The detector is update-capable if and only if causal=True.


    References
    ----------
    .. [1] Hampel F. R., "The influence curve and its role in robust estimation",
       Journal of the American Statistical Association, 69, 382-393, 1974


    Examples
    --------
    >>> from sktime.detection.hampel import HampelDetector
    >>> detector = HampelDetector()
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> detector.fit(y)
    HampelDetector()
    >>> y_anomalies = detector.predict(y)
    """

    _tags = {
        "authors": "SpitFire19",
        "maintainers": "SpitFire19",
        "python_dependencies": None,
        "object_type": "detector",
        "learning_type": "unsupervised",
        "task": "anomaly_detection",
        "capability:multivariate": False,
        "capability:missing_values": False,
        "capability:update": True,
        "capability:variable_identification": False,
        "X_inner_mtype": "pd.Series",
        "fit_is_empty": False,
    }

    def __init__(self, window_length=10, n_sigma=3.0, k=1.4826, causal=False):
        self.window_length = window_length
        self.n_sigma = n_sigma
        self.k = k
        self._buffer = pd.Series(dtype=float)
        self.causal = causal
        super().__init__()

    def _fit(self, X, y=None):
        """
        Initialize rolling buffer.

        Store only the last self.window_length points.
        """
        self._buffer = X.iloc[-self.window_length :].copy()
        return self

    def _update(self, X, y=None):
        """
        Update rolling buffer with new observations.

        Cut the historical data to store only one window of past data.
        """
        self._buffer = pd.concat([self._buffer, X]).iloc[-self.window_length :]
        return self

    def _get_window(self, x_values, i, history=None):
        """Extract centered or trailing window in function of self.causal."""
        # online mode
        if self.causal:
            combined = np.concatenate([history, [x_values[i]]])
            window_data = combined[-self.window_length :]
            new_history = window_data
            return window_data, new_history

        # offline mode
        hw = self.window_length // 2
        start = max(0, i - hw)
        end = min(len(x_values), i + hw + 1)
        window_data = x_values[start:end]
        return window_data, history

    def _predict(self, X):
        """
        Core logic for detecting outliers using a sliding window.

        Returns
        -------
        y : pd.Series
            Sparse series containing the iloc indices of detected anomalies.
        """
        x_values = X.values
        outlier_indices = []
        history = self._buffer.values if self.causal else np.array([])

        for i, x in enumerate(x_values):
            window_data, history = self._get_window(
                x_values,
                i,
                history,
            )
            median = np.nanmedian(window_data)
            mad = np.nanmedian(np.abs(window_data - median))
            sigma_est = max(self.k * mad, 1e-10)
            if np.abs(x - median) > self.n_sigma * sigma_est:
                outlier_indices.append(i)

        # Return a sparse pd.Series of outlier indices
        return pd.Series(outlier_indices, name="ilocs", dtype="int64")

    def _transform_scores(self, X):
        """
        Calculate the anomaly scores.

        Use a trailing or centered window depending on self.causal.

        Anomaly score is the absolute deviation normalized by (k * MAD).
        """
        x_values = X.values
        n = len(x_values)
        scores = np.zeros(n)
        history = self._buffer.values if self.causal else np.array([])

        for i, x in enumerate(x_values):
            # keep a trailing window of data
            window_data, history = self._get_window(
                x_values,
                i,
                history,
            )
            median = np.nanmedian(window_data)
            mad = np.nanmedian(np.abs(window_data - median))
            denom = max(self.k * mad, 1e-10)
            scores[i] = np.abs(x - median) / denom

        # Return a pd.Series of anomaly scores for the points of X
        return pd.DataFrame(scores, index=X.index, columns=["scores"])

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
        param1 = {"window_length": 20}
        param2 = {}
        param3 = {"window_length": 5, "n_sigma": 3.1, "k": 1.5}
        param4 = {"n_sigma": 4}
        param5 = {
            "window_length": 5,
            "n_sigma": 3.1,
            "k": 1.5,
            "causal": True,
        }
        return [param1, param2, param3, param4, param5]
