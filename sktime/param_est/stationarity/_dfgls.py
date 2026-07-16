# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimator for stationarity via DFGLS test from arch package."""

__author__ = ["Vasudeva-bit"]
__all__ = ["StationarityDFGLS"]

from sktime.param_est.base import BaseParamFitter


class StationarityDFGLS(BaseParamFitter):
    """Test for stationarity via the Dickey-Fuller GLS (DFGLS) Unit Root Test.

    Direct interface to ``DFGLS`` test from the ``arch`` package.

    Uses ``arch.unitroot.DFGLS`` as a test for unit roots,
    and derives a boolean statement whether a series is stationary.

    Also returns test results for the unit root test as fitted parameters.

    Parameters
    ----------
    lags : int, optional
        The number of lags to use in the ADF regression.  If omitted or None,
        ``method`` is used to automatically select the lag length with no more
        than ``max_lags`` are included.
    trend : {"c", "ct"}, optional
        The trend component to include in the test

        - "c" - Include a constant (Default)
        - "ct" - Include a constant and linear time trend

    max_lags : int, optional
        The maximum number of lags to use when selecting lag length. When using
        automatic lag length selection, the lag is selected using OLS
        detrending rather than GLS detrending ([2]_).
    method : {"AIC", "BIC", "t-stat"}, optional
        The method to use when selecting the lag length

        - "AIC" - Select the minimum of the Akaike IC
        - "BIC" - Select the minimum of the Schwarz/Bayesian IC
        - "t-stat" - Select the minimum of the Schwarz/Bayesian IC

    Attributes
    ----------
    stationary_ : bool
        whether the series in ``fit`` is stationary according to the test
        more precisely, whether the null of the Dickey-Fuller-GLS test is rejected at
        ``p_threshold``
    test_statistic_ : float
        The DFGLS test statistic, of running ``DFGLS`` on ``y`` in ``fit``
    pvalue_ : float : float
        p-value obtained when running ``DFGLS`` on ``y`` in ``fit``
    usedlag_ : int
        The number of lags used in the test.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.stationarity import StationarityDFGLS
    >>>
    >>> X = load_airline()  # doctest: +SKIP
    >>> sty_est = StationarityDFGLS()  # doctest: +SKIP
    >>> sty_est.fit(X)  # doctest: +SKIP
    StationarityDFGLS(...)
    >>> sty_est.get_fitted_params()["stationary"]  # doctest: +SKIP
    False
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["bashtage", "Vasudeva-bit"],  # bashtage for arch package
        "maintainers": "Vasudeva-bit",
        "python_dependencies": "arch",
        # estimator type
        # --------------
        "X_inner_mtype": ["pd.Series", "np.ndarray"],
        "scitype:X": "Series",
    }

    def __init__(
        self,
        lags=None,
        trend="c",
        max_lags=None,
        method="aic",
        p_threshold=0.05,
    ):
        self.lags = lags
        self.trend = trend
        self.max_lags = max_lags
        self.method = method
        self.p_threshold = p_threshold
        super().__init__()

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : {ndarray, Series}
        The data to test for a unit root

        Returns
        -------
        self : reference to self
        """
        from arch.unitroot import DFGLS

        p_threshold = self.p_threshold

        result = DFGLS(
            y=X,
            lags=self.lags,
            trend=self.trend,
            max_lags=self.max_lags,
            method=self.method,
        )
        self.test_statistic_ = result.stat
        self.pvalue = result.pvalue
        self.stationary_ = result.pvalue <= p_threshold
        self.used_lag_ = result._lags

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are no reserved values for parameter estimators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {
            "lags": 5,
            "trend": "ct",
            "max_lags": 10,
            "method": "t-stat",
            "p_threshold": 0.1,
        }

        return [params1, params2]
