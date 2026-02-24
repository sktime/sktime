"""Hampel filter for anomaly detection.

This implementation optionally supports a modified MAD (mMAD) variant that
approximates the standard MAD using a secondary median filter over absolute
deviations, improving computational efficiency while retaining robustness.
See: https://www.mdpi.com/1424-8220/25/11/3319
"""

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector

__author__ = ["Vbhatt03"]
__all__ = ["HampelFilter"]


class HampelFilter(BaseDetector):
    """Hampel filter for anomaly detection.

    Parameters
    ----------
    window_size : int, default=7
        Window size used to compute the rolling median and deviations. Must be >= 3. If
        ``center=True``, an even ``window_size`` is incremented to the next odd
        value.
    n_sigmas : float, default=3.0
        Threshold in scaled MAD units for flagging anomalies.
    center : bool, default=True
        If True, use a symmetric window around each time point. If False, use a
        trailing window suitable for causal detection.
    use_mmad : bool, default=False
        If True, compute the modified MAD (mMAD) by applying a secondary rolling
        median to the absolute deviations, optionally using a different window
        size controlled by ``mmad_window``. If False, the MAD is computed with
        the same ``window_size`` used for the central median.
    mmad_window : int, optional (default=None)
        Window size used for the mMAD median filter. If None, defaults to
        ``window_size``.

    Examples
    --------
    Detect anomalies in a univariate time series.

     >>> import pandas as pd
     >>> from sktime.detection.hampel import HampelFilter
     >>> y = pd.Series([1.0, 1.1, 0.9, 10.0, 1.0, 1.2, 0.8])
     >>> detector = HampelFilter(window_size=3, n_sigmas=3.0, center=True)
     >>> detector.fit(y)
     HampelFilter(...)
     >>> anomalies = detector.predict(y)
     >>> anomalies
         ilocs
     0      3
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
        if not use_mmad and mmad_window is not None:
            raise ValueError("mmad_window requires use_mmad=True")

        if n_sigmas <= 0:
            raise ValueError("n_sigmas must be > 0")
        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.center = center
        self.use_mmad = use_mmad
        if center and mmad_window is not None and mmad_window % 2 == 0:
            self.mmad_window = mmad_window + 1
        else:
            self.mmad_window = mmad_window
        super().__init__()

    def _predict(self, X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(
                    "HampelFilter only supports univariate input. "
                    f"Received a DataFrame with {X.shape[1]} columns."
                )
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
        mask = outliers.fillna(False)
        ilocs = np.where(mask)[0].tolist()
        return pd.DataFrame({"ilocs": ilocs})

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return. Currently only
            ``"default"`` is supported.

        Returns
        -------
        list of dict
            A list of parameter dictionaries to create test instances of
            the estimator.
        """
        params0 = {"window_size": 7, "n_sigmas": 3.0, "center": True}
        params1 = {"window_size": 9, "n_sigmas": 2.5, "center": False}
        return [params0, params1]
