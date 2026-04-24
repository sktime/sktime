"""Moving average anomaly detector."""

# References: #6481 (detection module wishlist)
# Related: Bollinger Bands concept (sktime/transformations/series/bollinger.py)

__author__ = ["rupeshca007"]

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector


class MovingAverageDetector(BaseDetector):
    """Anomaly detector based on a rolling mean and standard deviation threshold.

    Flags a time point as anomalous when the observed value deviates from the
    rolling mean by more than ``n_sigma`` standard deviations:

        |x_t - rolling_mean_t| > n_sigma * rolling_std_t

    This is a classical, interpretable baseline for anomaly detection in
    univariate time series, commonly used in industrial monitoring and
    sensor data quality checks.

    Parameters
    ----------
    window_size : int, default=10
        Size of the rolling window used to compute the mean and standard
        deviation. Must be a positive integer.
    n_sigma : float, default=3.0
        Number of standard deviations from the rolling mean that defines
        the detection threshold. Points outside the band
        [mean - n_sigma*std, mean + n_sigma*std] are flagged as anomalies.
        Must be a positive float.
    center : bool, default=False
        If True, the rolling window is centered around each time point
        (look-ahead is used). If False (default), only past values are used,
        which is causal and suitable for online/streaming detection.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.moving_average import MovingAverageDetector
    >>> X = pd.DataFrame(
    ...     [0.1, 0.2, 0.0, 0.15, 50.0, 0.1, -0.1, 0.05, 0.2, 0.0, 0.1, -60.0],
    ...     columns=["value"],
    ... )
    >>> detector = MovingAverageDetector(window_size=5, n_sigma=2.0)
    >>> detector.fit_transform(X)
       labels
    0       0
    1       0
    2       0
    3       0
    4       1
    5       0
    6       0
    7       0
    8       0
    9       0
    10      0
    11      1

    References
    ----------
    .. [1] Bollinger, J. (1992). Using Bollinger Bands. Stocks & Commodities, 10(2).
    .. [2] sktime detection module wishlist: https://github.com/sktime/sktime/issues/6481
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "rupeshca007",
        "maintainers": "rupeshca007",
        # estimator type
        # --------------
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "capability:multivariate": False,
        "fit_is_empty": True,
        # CI and test flags
        # -----------------
        "tests:core": True,
    }

    def __init__(self, window_size=10, n_sigma=3.0, center=False):
        self.window_size = window_size
        self.n_sigma = n_sigma
        self.center = center
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the detector (no-op for this stateless detector).

        Parameters
        ----------
        X : pd.DataFrame
            Time series data. Only univariate series are supported.
        y : ignored

        Returns
        -------
        self : reference to self
        """
        return self

    def _predict(self, X):
        """Detect anomalies in ``X`` using the rolling mean/std threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Univariate time series to detect anomalies in.

        Returns
        -------
        y_pred : pd.Series
            Sparse integer series of anomalous positional indices (0-based).
        """
        series = X.iloc[:, 0]

        # Use a SHIFTED (lagged) window so the current point is never included
        # in its own baseline computation. This prevents a spike from masking
        # itself by inflating its own rolling mean and std.
        if self.center:
            # For centered mode, still exclude the current point by using
            # a window that covers neighbors but shifts by 1 on each side.
            half = self.window_size // 2
            rolling_mean = series.shift(1).rolling(
                window=self.window_size,
                min_periods=1,
                center=False,
            ).mean()
            rolling_std = series.shift(1).rolling(
                window=self.window_size,
                min_periods=1,
                center=False,
            ).std(ddof=0)
        else:
            # Causal: use only the previous window_size points (shift by 1)
            shifted = series.shift(1)
            rolling_mean = shifted.rolling(
                window=self.window_size,
                min_periods=1,
            ).mean()
            rolling_std = shifted.rolling(
                window=self.window_size,
                min_periods=1,
            ).std(ddof=0)

        # Threshold: flag points where deviation > n_sigma * std.
        # Special case: if rolling_std == 0 (perfectly flat baseline), flag
        # any deviation at all as anomalous — the flat history is clearly broken.
        deviation = (series - rolling_mean).abs()
        threshold = self.n_sigma * rolling_std

        std_nonzero_anomaly = (rolling_std > 0) & (deviation > threshold)
        std_zero_anomaly = (rolling_std == 0) & (deviation > 0)
        anomaly_mask = std_nonzero_anomaly | std_zero_anomaly

        anomaly_positions = np.where(anomaly_mask.values)[0]

        if len(anomaly_positions) == 0:
            return self._empty_sparse()

        return pd.Series(anomaly_positions, dtype=int)

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
        params : dict or list of dict
            Parameters to create testing instances of the class.
            Each dict is a valid parameter set for ``__init__``.
        """
        params0 = {
            "window_size": 5,
            "n_sigma": 3.0,
            "center": False,
        }
        params1 = {
            "window_size": 10,
            "n_sigma": 2.0,
            "center": True,
        }
        return [params0, params1]
