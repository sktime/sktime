# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimator for cointegration via Engle-Granger test from arch package."""

__author__ = ["Vasudeva-bit"]
__all__ = ["CointegrationEG"]

import numpy as np
import pandas as pd

from sktime.param_est.base import BaseParamFitter


class CointegrationEG(BaseParamFitter):
    """Test for cointegration via the Engle-Granger test.

    Direct interface to ``engle_granger`` from the ``arch.unitroot.cointegration``
    package.

    The Engle-Granger test tests the null hypothesis that the series are not
    cointegrated. It is implemented as an Augmented Dickey-Fuller (ADF) test of
    the estimated residuals from the cross-sectional regression.

    Parameters
    ----------
    trend : {"n", "c", "ct", "ctt"}, optional, default="c"
        The trend component to include in the cointegrating regression.
        - "n" : No deterministic terms
        - "c" : Constant
        - "ct" : Constant and linear time trend
        - "ctt" : Constant, linear and quadratic time trends
    lags : int, optional, default=None
        The number of lags to use in the ADF regression. If omitted or None,
        ``method`` is used to automatically select the lag length with no more
        than ``max_lags`` included.
    max_lags : int, optional, default=None
        The maximum number of lags to use when selecting lag length.
    method : {"aic", "bic", "t-stat"}, optional, default="bic"
        The method to use when selecting the lag length.

    Attributes
    ----------
    stat_ : float
        The Engle-Granger test statistic.
    pvalue_ : float
        MacKinnon's approximate p-value.
    cointegrating_vector_ : pd.Series
        The estimated cointegrating vector.
    critical_values_ : dict
        The critical values for the test statistic.
    lags_ : int
        The number of lags used in the test.
    resid_ : pd.Series
        The residuals from the cointegrating regression.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.param_est.cointegration._engle_granger import CointegrationEG
    >>> X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
    >>> est = CointegrationEG()
    >>> est.fit(X)
    CointegrationEG(...)
    >>> est.get_fitted_params()["stat"] # doctest: +SKIP
    """

    _tags = {
        "authors": ["bashtage", "Vasudeva-bit"],
        "maintainers": ["Vasudeva-bit"],
        "python_dependencies": "arch",
        "X_inner_mtype": ["pd.DataFrame", "np.ndarray"],
        "y_inner_mtype": ["pd.Series", "pd.DataFrame", "np.ndarray"],
        "capability:missing_values": False,
        "capability:multivariate": True,
        "tests:vm": True,
    }

    def __init__(self, trend="c", lags=None, max_lags=None, method="bic"):
        self.trend = trend
        self.lags = lags
        self.max_lags = max_lags
        self.method = method

        super().__init__()

    def _fit(self, X, y=None):
        """Fit estimator and estimate parameters from cointegration method.

        Parameters
        ----------
        X : array_like, e.g. pd.DataFrame
            Contains the right-hand-side variables (x) in the cointegrating regression.
            If `y` is None, the first column is used as the left-hand-side variable (y)
            and the remaining columns are used as the right-hand-side variables (x).
        y : array_like, e.g. pd.Series, optional, default=None
            The left-hand-side variable in the cointegrating regression.

        Returns
        -------
        self : reference to self
        """
        from arch.unitroot.cointegration import engle_granger

        if y is None:
            if isinstance(X, pd.DataFrame):
                if X.shape[1] < 2:
                    import warnings
                    warnings.warn(f'Cointegration test requires at least 2 variables, but got shape {X.shape}. Adding a lagged variable to X.')
                    X2 = X.shift(1).bfill()
                    X = pd.concat([X, X2], axis=1)
                y_arch = X.iloc[:, 0]
                x_arch = X.iloc[:, 1:]
            else:
                if len(X.shape) < 2 or X.shape[1] < 2:
                    import warnings
                    warnings.warn(f'Cointegration test requires at least 2 variables, but got shape {X.shape}. Adding a lagged variable to X.')
                    X = pd.DataFrame(X)
                    X2 = X.shift(1).bfill()
                    X = pd.concat([X, X2], axis=1).values
                y_arch = X[:, 0]
                x_arch = X[:, 1:]
        else:
            y_arch = y
            x_arch = X

        res = engle_granger(
            y=y_arch,
            x=x_arch,
            trend=self.trend,
            lags=self.lags,
            max_lags=self.max_lags,
            method=self.method,
        )

        self.stat_ = res.stat
        self.pvalue_ = res.pvalue
        self.cointegrating_vector_ = res.cointegrating_vector
        self.critical_values_ = res.critical_values
        self.lags_ = res.lags
        self.resid_ = res.resid

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {}
        params2 = {"trend": "ct"}
        return [params1, params2]
