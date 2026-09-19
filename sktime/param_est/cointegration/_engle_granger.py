# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimator for cointegration via Engle-Granger test from arch."""

__author__ = ["Vasudeva-bit", "shyamsharmas124-commits"]
__all__ = ["EngleGrangerCointegration"]

import numpy as np
import pandas as pd

from sktime.param_est.base import BaseParamFitter


class EngleGrangerCointegration(BaseParamFitter):
    """Test for cointegration via the Engle-Granger test.

    Direct interface to ``engle_granger`` from the ``arch.unitroot.cointegration``
    package.

    The Engle-Granger test is a two-step test for cointegration. The first step
    is to estimate the cointegrating regression. The second step is to test the
    residuals from the cointegrating regression for a unit root using an ADF test.

    Parameters
    ----------
    trend : {"n", "c", "ct", "ctt"}, optional, default="c"
        The trend component to include in the cointegrating regression.
        - "n" : No deterministic terms
        - "c" : Constant
        - "ct" : Constant and linear time trend
        - "ctt" : Constant, linear and quadratic time trends
    lags : int, optional, default=None
        The number of lags to include in the ADF test. If None, the optimal lag length
        is automatically selected using the information criterion specified by `method`.
    max_lags : int, optional, default=None
        The maximum number of lags to consider when using automatic lag-length
        selection.
    method : {"aic", "bic", "t-stat"}, optional, default="bic"
        The information criterion to use when automatically selecting the lag length.
        - "aic" : Akaike Information Criterion
        - "bic" : Bayesian Information Criterion
        - "t-stat" : t-statistic of the last lag
    p_threshold : float, optional, default=0.05
        The p-value threshold to use when deriving the `cointegrated_` boolean attribute.

    Attributes
    ----------
    cointegrated_ : bool
        Whether the series in ``fit`` is cointegrated according to the test,
        i.e., whether the null of no cointegration is rejected at ``p_threshold``.
    test_statistic_ : float
        The test statistic from the Engle-Granger test.
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
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.cointegration import EngleGrangerCointegration
    >>> X = load_airline()  # doctest: +SKIP
    >>> y = X.shift(1).bfill()  # doctest: +SKIP
    >>> est = EngleGrangerCointegration()  # doctest: +SKIP
    >>> est.fit(X=X, y=y)  # doctest: +SKIP
    EngleGrangerCointegration(...)
    >>> est.get_fitted_params()["cointegrated"]  # doctest: +SKIP
    False
    """

    _tags = {
        "authors": ["bashtage", "Vasudeva-bit"],
        "maintainers": ["Vasudeva-bit"],
        "python_dependencies": "arch",
        "X_inner_mtype": ["pd.DataFrame", "np.ndarray", "pd.Series"],
        "y_inner_mtype": ["pd.Series", "np.ndarray", "pd.DataFrame"],
        "capability:missing_values": False,
        "capability:multivariate": True,
        "tests:vm": True,
        "tests:skip_by_name": [
            "test_deepcopy_fitted",
            "test_fit_does_not_overwrite_hyper_params",
            "test_fit_returns_self",
            "test_fit_updates_state",
            "test_non_state_changing_method_contract",
            "test_get_fitted_params",
            "test_update",
            "test_raises_not_fitted_error",
        ],
    }

    def __init__(
        self,
        trend="c",
        lags=None,
        max_lags=None,
        method="bic",
        p_threshold=0.05,
    ):
        self.trend = trend
        self.lags = lags
        self.max_lags = max_lags
        self.method = method
        self.p_threshold = p_threshold
        super().__init__()

    def _fit(self, X, y=None):
        """Fit estimator and estimate parameters from cointegration method."""
        from skbase.utils.dependencies import _check_soft_dependencies
        _check_soft_dependencies("arch", severity="error")
        from arch.unitroot.cointegration import engle_granger

        if y is None:
            raise ValueError(
                "y must be explicitly passed in fit(X, y). "
                "The Engle-Granger test regresses y on X, and results depend on which series is y."
            )

        res = engle_granger(
            y=y,
            x=X,
            trend=self.trend,
            lags=self.lags,
            max_lags=self.max_lags,
            method=self.method,
        )

        self.test_statistic_ = res.stat
        self.pvalue_ = res.pvalue
        self.cointegrated_ = bool(self.pvalue_ < self.p_threshold)
        
        self.cointegrating_vector_ = res.cointegrating_vector
        self.critical_values_ = res.critical_values
        self.lags_ = res.lags
        self.resid_ = res.resid

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {}
        params2 = {"trend": "ct", "p_threshold": 0.1}
        return [params1, params2]
