# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimator for stationarity via ADF test from arch package."""

__author__ = ["Vasudeva-bit"]
__all__ = ["StationarityADFArch"]

from sktime.param_est.base import BaseParamFitter


class StationarityADFArch(BaseParamFitter):
    """Test for stationarity via the Augmented Dickey-Fuller Unit Root Test (ADF).

    Direct interface to ``DFGLS`` test from the ``arch`` package.
    Does not assume ARCH process, naming is due to the use of the ``arch`` package.

    Uses ``arch.unitroot.ADF`` as a test for unit roots,
    and derives a boolean statement whether a series is stationary.

    Also returns test results for the unit root test as fitted parameters.

    Parameters
    ----------
    lags : int, optional
        The number of lags to use in the ADF regression.  If omitted or None,
        ``method`` is used to automatically select the lag length with no more
        than ``max_lags`` are included.
    trend : {"n", "c", "ct", "ctt"}, optional
        The trend component to include in the test

        - "n" - No trend components
        - "c" - Include a constant (Default)
        - "ct" - Include a constant and linear time trend
        - "ctt" - Include a constant and linear and quadratic time trends

    max_lags : int, optional
        The maximum number of lags to use when selecting lag length
    method : {"AIC", "BIC", "t-stat"}, optional
        The method to use when selecting the lag length

        - "AIC" - Select the minimum of the Akaike IC
        - "BIC" - Select the minimum of the Schwarz/Bayesian IC
        - "t-stat" - Select the minimum of the Schwarz/Bayesian IC

    low_memory : bool
        Flag indicating whether to use a low memory implementation of the
        lag selection algorithm. The low memory algorithm is slower than
        the standard algorithm but will use 2-4% of the memory required for
        the standard algorithm. This options allows automatic lag selection
        to be used in very long time series. If None, use automatic selection
        of algorithm.

    Attributes
    ----------
    stationary_ : bool
        whether the series in ``fit`` is stationary according to the test
        more precisely, whether the null of the ADF test is rejected at ``p_threshold``
    test_statistic_ : float
        The ADF test statistic, of running ``adfuller`` on ``y`` in ``fit``
    pvalue_ : float : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010),
        obtained when running ``adfuller`` on ``y`` in ``fit``
    usedlag_ : int
        The number of lags used in the test.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.stationarity import StationarityADFArch
    >>>
    >>> X = load_airline()  # doctest: +SKIP
    >>> sty_est = StationarityADFArch()  # doctest: +SKIP
    >>> sty_est.fit(X)  # doctest: +SKIP
    StationarityADFArch(...)
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
        low_memory=None,
        p_threshold=0.05,
    ):
        self.lags = lags
        self.trend = trend
        self.max_lags = max_lags
        self.method = method
        self.low_memory = low_memory
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
        from arch.unitroot import ADF

        p_threshold = self.p_threshold

        result = ADF(
            y=X,
            lags=self.lags,
            trend=self.trend,
            max_lags=self.max_lags,
            method=self.method,
            low_memory=self.low_memory,
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
            "trend": "ctt",
            "max_lags": 10,
            "method": "t-stat",
            "low_memory": True,
            "p_threshold": 0.1,
        }

        return [params1, params2]
