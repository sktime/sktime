"""Hampel Filter for outlier detection in time series."""

__author__ = ["Saurabh6266"]

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector


class HampelFilter(BaseDetector):
    """Hampel Filter for outlier/anomaly detection in univariate time series.

    Identifies outliers by computing, for each data point, the median and
    scaled Median Absolute Deviation (MAD) within a sliding window. A point
    is flagged as an outlier if it deviates from the local median by more than
    ``n_sigma`` times the scaled MAD.

    The MAD is scaled by the factor 1.4826 so that it is a consistent
    estimator of the standard deviation for normally distributed data.

    Parameters
    ----------
    window_size : int, default=5
        Half-width of the sliding window. The total window considered for
        each point ``i`` spans indices ``[i - window_size, i + window_size]``
        (clipped to series boundaries). Must be a positive integer.
    n_sigma : float, default=3.0
        Threshold multiplier applied to the scaled MAD. Points deviating by
        more than ``n_sigma * 1.4826 * MAD`` from the local median are
        flagged as outliers. Typical values: 3.0 (conservative), 2.0 (aggressive).

    Attributes
    ----------
    is_fitted_ : bool
        True after ``fit`` is called. Since the Hampel Filter is fully
        unsupervised, fitting is a no-op.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.hampel import HampelFilter
    >>> X = pd.Series([1.0, 1.2, 1.1, 50.0, 1.3, 1.0, 1.1, 1.2])
    >>> detector = HampelFilter(window_size=3, n_sigma=3.0)
    >>> detector.fit(X)
    HampelFilter(...)
    >>> detector.predict(X)
       ilocs
    0      3

    References
    ----------
    .. [1] Hampel, F. R. (1974). The influence curve and its role in robust
       estimation. Journal of the American Statistical Association, 69(346),
       383-393.
    .. [2] Liu, H., Shah, S., & Jiang, W. (2004). On-line outlier detection
       and data cleaning. Computers & Chemical Engineering, 28(9), 1635-1647.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "Saurabh6266",
        "maintainers": "Saurabh6266",
        # estimator type
        # --------------
        "fit_is_empty": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "X_inner_mtype": "pd.Series",
    }

    def __init__(self, window_size: int = 5, n_sigma: float = 3.0):
        self.window_size = window_size
        self.n_sigma = n_sigma
        super().__init__()

    def _fit(self, X: pd.Series, y=None):
        """Fit the Hampel Filter.

        The Hampel Filter is unsupervised — fitting is a no-op. This method
        exists to conform to the sktime estimator interface.

        Parameters
        ----------
        X : pd.Series
            Time series on which to fit. Ignored.
        y : ignored

        Returns
        -------
        self : HampelFilter
            Reference to self.
        """
        return self

    def _predict(self, X: pd.Series) -> pd.Series:
        """Detect outliers in a time series using the Hampel Filter.

        For each point, computes the local median and MAD over a sliding
        window of half-width ``window_size``. Points deviating by more than
        ``n_sigma * 1.4826 * MAD`` from the local median are flagged.

        When MAD is zero (constant window), any point differing from the
        window median is flagged as an outlier.

        Parameters
        ----------
        X : pd.Series
            Univariate time series to detect outliers in. Must not contain
            NaN values (see ``capability:missing_values`` tag).

        Returns
        -------
        pd.Series
            Integer positions (ilocs) of detected outliers. The base class
            converts this into a sparse DataFrame with column ``"ilocs"``.
        """
        # 1.4826 makes MAD a consistent estimator of std for Gaussian data
        _SCALE = 1.4826

        values = X.to_numpy(dtype=float)
        n = len(values)
        outlier_ilocs = []

        for i in range(n):
            lo = max(0, i - self.window_size)
            hi = min(n, i + self.window_size + 1)
            window = values[lo:hi]

            median = np.median(window)
            mad = np.median(np.abs(window - median))

            if mad == 0.0:
                # Constant window — flag only if this point differs from it
                if values[i] != median:
                    outlier_ilocs.append(i)
            else:
                threshold = self.n_sigma * _SCALE * mad
                if abs(values[i] - median) > threshold:
                    outlier_ilocs.append(i)

        return pd.Series(outlier_ilocs, dtype="int64")

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        list of dict
            Each dict is a valid set of constructor parameters.
        """
        params1 = {"window_size": 3, "n_sigma": 3.0}
        params2 = {"window_size": 5, "n_sigma": 2.5}
        return [params1, params2]
    