"""Hampel filter for anomaly detection.

This implementation optionally supports a modified MAD (mMAD) variant that
approximates the standard MAD using a secondary median filter over absolute
deviations, improving computational efficiency while retaining robustness.
See: https://www.mdpi.com/1424-8220/25/11/3319
"""

import pandas as pd

from sktime.detection.base import BaseDetector

__author__ = ["Vbhatt03"]
__all__ = ["HampelFilter"]


class HampelFilter(BaseDetector):
    """Hampel filter for anomaly detection.

    Parameters
    ----------
    window_size : int, default=7
        Window size used to compute the rolling median and deviations. If
        ``center=True``, an even ``window_size`` is incremented to the next odd
        value.
    n_sigmas : float, default=3.0
        Threshold in scaled MAD units for flagging anomalies.
    center : bool, default=True
        If True, use a symmetric window around each time point. If False, use a
        trailing window suitable for causal detection.
    use_mmad : bool, default=False
        If True, compute the modified MAD (mMAD) by applying a median filter to
        the absolute deviations instead of computing the per-window MAD.
    mmad_window : int, optional (default=None)
        Window size used for the mMAD median filter. If None, defaults to
        ``window_size``.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "Vbhatt03",
        "maintainers": "Vbhatt03",
        # estimator type
        # --------------
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "capability:multivariate": False,
        "fit_is_empty": True,
        "X_inner_mtype": "pd.Series",
        # CI and test flags
        # -----------------
        "tests:core": True,
    }

    def __init__(
        self, window_size=7, n_sigmas=3.0, center=True, use_mmad=False, mmad_window=None
    ):
        # validate window_size
        if window_size < 3:
            raise ValueError("window_size must be >= 3")
        if center and window_size % 2 == 0:
            window_size += 1

        if use_mmad and mmad_window is not None and mmad_window < 3:
            raise ValueError("mmad_window must be >= 3")

        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.center = center
        self.use_mmad = use_mmad
        self.mmad_window = mmad_window
        super().__init__()

    def _predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]

        median_series = X.rolling(self.window_size, center=self.center).median()
        abs_dev = (X - median_series).abs()

        if self.use_mmad:
            mmad_window = self.mmad_window or self.window_size
            mad_series = abs_dev.rolling(mmad_window, center=self.center).median()
        else:
            mad_series = abs_dev.rolling(self.window_size, center=self.center).median()

        sigma = 1.4826 * mad_series
        outliers = abs_dev > (self.n_sigmas * sigma)

        # extract indices
        outlier_indices = X.index[outliers.fillna(False)]
        return pd.DataFrame({"ilocs": outlier_indices})

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params0 = {"window_size": 7, "n_sigmas": 3.0, "center": True}
        params1 = {"window_size": 9, "n_sigmas": 2.5, "center": False}
        return [params0, params1]
