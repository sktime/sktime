# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimators for stationarity."""

__author__ = ["fkiraly"]
__all__ = [
    "StationarityADF",
    "StationarityKPSS",
]

from sktime.param_est.base import BaseParamFitter


class StationarityADF(BaseParamFitter):
    """Test for stationarity via the Augmented Dickey-Fuller Unit Root Test (ADF).

    Uses ``statsmodels.tsa.stattools.adfuller`` as a test for unit roots,
    and derives a boolean statement whether a series is stationary.

    Also returns test results for the unit root test as fitted parameters.

    Parameters
    ----------
    p_threshold : float, optional, default=0.05
        significance threshold to apply in testing for stationarity
    maxlag : int or None, optional, default=None
        Maximum lag which is included in test, default value of
        12*(nobs/100)^{1/4} is used when ``None``.
    regression : str, one of {"c","ct","ctt","n"}, optional, default="c"
        Constant and trend order to include in regression.

        * "c" : constant only (default).
        * "ct" : constant and trend.
        * "ctt" : constant, and linear and quadratic trend.
        * "n" : no constant, no trend.

    autolag : one of {"AIC", "BIC", "t-stat", None}, optional, default="AIC"
        Method to use when automatically determining the lag length among the
        values 0, 1, ..., maxlag.

        * If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
        * If None, then the number of included lags is set to maxlag.

    Attributes
    ----------
    stationary_ : bool
        whether the series in ``fit`` is stationary according to the test
        more precisely, whether the null of the ADF test is rejected at ``p_threshold``
    test_statistic_ : float
        The ADF test statistic, of running ``adfuller`` on ``y`` in ``fit``
    pvalue_ : float : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010),
        obtained when running `adfuller` on ``y`` in ``fit``
    usedlag_ : int
        The number of lags used in the test.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.stationarity import StationarityADF
    >>>
    >>> X = load_airline()  # doctest: +SKIP
    >>> sty_est = StationarityADF()  # doctest: +SKIP
    >>> sty_est.fit(X)  # doctest: +SKIP
    StationarityADF(...)
    >>> sty_est.get_fitted_params()["stationary"]  # doctest: +SKIP
    False
    """

    _tags = {
        "X_inner_mtype": "pd.Series",  # which types do _fit/_predict, support for X?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "capability:missing_values": False,  # can estimator handle missing data?
        "capability:multivariate": False,  # can estimator handle multivariate data?
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        p_threshold=0.05,
        maxlag=None,
        regression="c",
        autolag="AIC",
    ):
        self.p_threshold = p_threshold
        self.maxlag = maxlag
        self.regression = regression
        self.autolag = autolag
        super().__init__()

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.

        Returns
        -------
        self : reference to self
        """
        from statsmodels.tsa.stattools import adfuller

        p_threshold = self.p_threshold

        res = adfuller(
            x=X,
            maxlag=self.maxlag,
            regression=self.regression,
            autolag=self.autolag,
        )
        self.test_statistic_ = res[0]
        self.pvalue_ = res[1]
        self.stationary_ = res[1] <= p_threshold
        self.used_lag_ = res[2]

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {
            "p_threshold": 0.1,
            "maxlag": 5,
            "regression": "ctt",
            "autolag": "t-stat",
        }

        return [params1, params2]


class StationarityKPSS(BaseParamFitter):
    """Test for stationarity via the Kwiatkowski-Phillips-Schmidt-Shin Test.

    Uses ``statsmodels.tsa.stattools.kpss`` as a test for trend-stationairty,
    and derives a boolean statement whether a series is (trend-)stationary.

    Also returns test results for the trend-stationarity test as fitted parameters.

    Parameters
    ----------
    p_threshold : float, optional, default=0.05
        significance threshold to apply in testing for stationarity
    regression : str, one of {"c","ct","ctt","n"}, optional, default="c"
        Constant and trend order to include in regression.

        * "c" : constant only (default).
        * "ct" : constant and trend.
        * "ctt" : constant, and linear and quadratic trend.
        * "n" : no constant, no trend.

    nlags : str or int, optional, default="auto". If int, must be positive.
        Indicates the number of lags to be used internally in `kpss`.
        If "auto", lags is calculated using the data-dependent method of Hobijn et al
        (1998). See also Andrews (1991), Newey & West (1994), and Schwert (1989).
        If "legacy", uses int(12 * (n / 100)**(1 / 4)) , as outlined in Schwert (1989).
        If int, uses that exact number.

    Attributes
    ----------
    stationary_ : bool
        whether the series in ``fit`` is stationary according to the test
        more precisely, whether the null of the KPSS test is accepted at ``p_threshold``
    test_statistic_ : float
        The KPSS test statistic, of running ``kpss`` on ``y`` in ``fit``
    pvalue_ : float : float
        The p-value of the KPSS test, of running ``kpss`` on ``y`` in ``fit``.
        The p-value is interpolated from Table 1 in Kwiatkowski et al. (1992),
        and a boundary point is returned if the test statistic is outside the table of
        critical values, that is, if the p-value is outside the interval (0.01, 0.1).
    lags_ : int
        The truncation lag parameter.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.stationarity import StationarityKPSS
    >>>
    >>> X = load_airline()  # doctest: +SKIP
    >>> sty_est = StationarityKPSS()  # doctest: +SKIP
    >>> sty_est.fit(X)  # doctest: +SKIP
    StationarityKPSS(...)
    >>> sty_est.get_fitted_params()["stationary"]  # doctest: +SKIP
    False
    """

    _tags = {
        "X_inner_mtype": "pd.Series",  # which types do _fit/_predict, support for X?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "capability:missing_values": False,  # can estimator handle missing data?
        "capability:multivariate": False,  # can estimator handle multivariate data?
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        p_threshold=0.05,
        regression="c",
        nlags="auto",
    ):
        self.p_threshold = p_threshold
        self.regression = regression
        self.nlags = nlags
        super().__init__()

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.

        Returns
        -------
        self : reference to self
        """
        from statsmodels.tsa.stattools import kpss

        p_threshold = self.p_threshold

        res = kpss(
            x=X,
            regression=self.regression,
            nlags=self.nlags,
        )
        self.test_statistic_ = res[0]
        self.pvalue_ = res[1]
        self.stationary_ = res[1] > p_threshold
        self.lags_ = res[2]

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"p_threshold": 0.1, "regression": "ctt", "nlags": 5}

        return [params1, params2]
