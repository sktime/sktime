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
    >>> anomaly_scores = detector.transform_scores(y)
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
        "capability:update": False,
        "capability:variable_identification": False,
        "X_inner_mtype": "pd.Series",
        "fit_is_empty": True,
    }

    def __init__(self, window_length=10, n_sigma=3.0, k=1.4826):
        self.window_length = window_length
        self.n_sigma = n_sigma
        self.k = k
        super().__init__()

    def _predict(self, X):
        """
        Core logic for detecting outliers using a sliding window.

        Returns
        -------
        y : pd.Series
            Sparse series containing the iloc indices of detected anomalies.
        """
        x_values = X.values
        n = len(x_values)
        outlier_indices = []

        # Half-window for a centered approach
        hw = self.window_length // 2

        for i in range(n):
            start = max(0, i - hw)
            end = min(n, i + hw + 1)

            window_data = x_values[start:end]

            # Calculate outlier statistics
            median = np.nanmedian(window_data)
            mad = np.nanmedian(np.abs(window_data - median))

            sigma_est = self.k * mad

            # Compare absolute deviation to the threshold
            if np.abs(x_values[i] - median) > self.n_sigma * sigma_est:
                outlier_indices.append(i)

        # Return a series that BaseDetector will coerce to the 'ilocs' format
        return pd.Series(outlier_indices, name="ilocs", dtype="int64")

    def _transform_scores(self, X):
        """
        Calculate the anomaly scores (deviation in terms of n * sigma_est).

        Anomaly score is the absolute deviation normalized by (k * MAD).
        """
        x_values = X.values
        n = len(x_values)
        scores = np.zeros(n)
        hw = self.window_length // 2

        for i in range(n):
            # Ensure valid start and end indices for the window
            start = max(0, i - hw)
            end = min(n, i + hw + 1)
            window_data = x_values[start:end]

            median = np.nanmedian(window_data)
            mad = np.nanmedian(np.abs(window_data - median))
            # Ensure the denominator is non-zero
            denom = np.maximum(self.k * mad, 1e-9)
            scores[i] = np.abs(x_values[i] - median) / denom

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
        return [param1, param2, param3, param4]
