# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimators for stationarity."""

__author__ = ["Vasudeva-bit"]
__all__ = [
    "StationarityADFArch",
    "StationarityDFGLS",
    "StationarityPhillipsPerron",
    "StationarityKPSSArch",
    "StationarityZivotAndrews",
    "StationarityVarianceRatio",
]

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
        "X_inner_mtype": ["pd.Series", "nd.array"],
        "scitype:X": "Series",
        "python_dependencies": "arch",
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
            special parameters are defined for a value, will return `"default"` set.
            There are no reserved values for parameter estimators.

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
            "lags": 5,
            "trend": "ctt",
            "max_lags": 10,
            "method": "t-stat",
            "low_memory": True,
            "p_threshold": 0.1,
        }

        return [params1, params2]


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
        "X_inner_mtype": ["pd.Series", "nd.array"],
        "scitype:X": "Series",
        "python_dependencies": "arch",
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
            special parameters are defined for a value, will return `"default"` set.
            There are no reserved values for parameter estimators.

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
            "lags": 5,
            "trend": "ct",
            "max_lags": 10,
            "method": "t-stat",
            "p_threshold": 0.1,
        }

        return [params1, params2]


class StationarityPhillipsPerron(BaseParamFitter):
    """Test for unit root order 1 via the Phillips-Perron Unit Root Test.

    Direct interface to ``PhillipsPerron`` test from the ``arch`` package.

    Uses ``arch.unitroot.PhillipsPerron`` as a test for unit roots,
    and derives a boolean statement whether a series is stationary.

    Also returns test results for the unit root test as fitted parameters.

    Parameters
    ----------
    lags : int, optional
        The number of lags to use in the Newey-West estimator of the long-run
        covariance. If omitted or None, the lag length is set automatically to
        ``12 * (nobs/100) ** (1/4)``
    trend : {"n", "c", "ct"}, optional
        The trend component to include in the test

        - "n" - No trend components
        - "c" - Include a constant (Default)
        - "ct" - Include a constant and linear time trend

    test_type : {"tau", "rho"}
        The test to use when computing the test statistic. "tau" is based on
        the t-stat and "rho" uses a test based on nobs times the re-centered
        regression coefficient

    Attributes
    ----------
    stationary_ : bool
        whether the series in ``fit`` is integrated of order 1
        more precisely, whether the null of the Phillips-Perron test is rejected at
        ``p_threshold``
    test_statistic_ : float
        The PP test statistic, of running ``PhillipsPerron`` on ``y`` in ``fit``
    pvalue_ : float : float
        p-value obtained when running ``PhillipsPerron`` on ``y`` in ``fit``
    usedlag_ : int
        The number of lags used in the test.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.stationarity import StationarityPhillipsPerron
    >>>
    >>> X = load_airline()  # doctest: +SKIP
    >>> sty_est = StationarityPhillipsPerron()  # doctest: +SKIP
    >>> sty_est.fit(X)  # doctest: +SKIP
    StationarityPhillipsPerron(...)
    >>> sty_est.get_fitted_params()["stationary"]  # doctest: +SKIP
    False
    """

    _tags = {
        "X_inner_mtype": ["pd.Series", "nd.array"],
        "scitype:X": "Series",
        "python_dependencies": "arch",
    }

    def __init__(
        self,
        lags=None,
        trend="c",
        test_type="tau",
        p_threshold=0.05,
    ):
        self.lags = lags
        self.trend = trend
        self.test_type = test_type
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
        from arch.unitroot import PhillipsPerron

        p_threshold = self.p_threshold

        result = PhillipsPerron(
            y=X,
            lags=self.lags,
            trend=self.trend,
            test_type=self.test_type,
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
            special parameters are defined for a value, will return `"default"` set.
            There are no reserved values for parameter estimators.

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
            "lags": 5,
            "trend": "ct",
            "test_type": "rho",
            "p_threshold": 0.1,
        }

        return [params1, params2]


class StationarityKPSSArch(BaseParamFitter):
    """Test for stationarity via the Kwiatkowski-Phillips-Schmidt-Shin Unit Root Test.

    Direct interface to ``KPSS`` test from the ``arch`` package.
    Does not assume ARCH process, naming is due to the use of the ``arch`` package.

    Uses ``arch.unitroot.KPSS`` as a test for trend-stationarity,
    and derives a boolean statement whether a series is (trend-)stationary.

    Also returns test results for the unit root test as fitted parameters.

    Parameters
    ----------
    lags : int, optional
        The number of lags to use in the Newey-West estimator of the long-run
        covariance.  If omitted or None, the number of lags is calculated
        with the data-dependent method of Hobijn et al. (1998). See also
        Andrews (1991), Newey & West (1994), and Schwert (1989).
        Set ``lags=-1`` to use the old method that only depends on the sample
        size, ``12 * (nobs/100) ** (1/4)``.
    trend : {"c", "ct"}, optional
        The trend component to include in the ADF test
            "c" - Include a constant (Default)
            "ct" - Include a constant and linear time trend

    Attributes
    ----------
    stationary_ : bool
        whether the series in ``fit`` is stationary according to the test
        more precisely, whether the null of the KPSS test is accepted at ``p_threshold``
    test_statistic_ : float
        The KPSS test statistic, of running ``KPSS`` on ``y`` in ``fit``
    pvalue_ : float : float
        p-value obtained when running ``KPSS`` on ``y`` in ``fit``
    usedlag_ : int
        The number of lags used in the test.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.stationarity import StationarityKPSSArch
    >>>
    >>> X = load_airline()  # doctest: +SKIP
    >>> sty_est = StationarityKPSSArch()  # doctest: +SKIP
    >>> sty_est.fit(X)  # doctest: +SKIP
    StationarityKPSSArch(...)
    >>> sty_est.get_fitted_params()["stationary"]  # doctest: +SKIP
    True
    """

    _tags = {
        "X_inner_mtype": ["pd.Series", "nd.array"],
        "scitype:X": "Series",
        "python_dependencies": "arch",
    }

    def __init__(
        self,
        lags=None,
        trend="c",
        p_threshold=0.05,
    ):
        self.lags = lags
        self.trend = trend
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
        from arch.unitroot import KPSS

        p_threshold = self.p_threshold

        result = KPSS(
            y=X,
            lags=self.lags,
            trend=self.trend,
        )
        self.test_statistic_ = result.stat
        self.pvalue = result.pvalue
        self.stationary_ = result.pvalue > p_threshold
        self.used_lag_ = result._lags

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are no reserved values for parameter estimators.

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
            "lags": 5,
            "trend": "ct",
            "p_threshold": 0.1,
        }

        return [params1, params2]


class StationarityZivotAndrews(BaseParamFitter):
    """Test for stationarity via the Zivot-Andrews Unit Root Test.

    Direct interface to ``ZivotAndrews`` test from the `arch` package.

    Uses ``arch.unitroot.ZivotAndrews`` as a test for unit roots,
    and derives a boolean statement whether a series is stationary.

    Also returns test results for the unit root test as fitted parameters.

    Parameters
    ----------
    lags : int, optional
        The number of lags to use in the ADF regression. If omitted or None,
        ``method`` is used to automatically select the lag length with no more
        than `max_lags` are included.
    trend : {"c", "t", "ct"}, optional
        The trend component to include in the test

        - "c" - Include a constant (Default)
        - "t" - Include a linear time trend
        - "ct" - Include a constant and linear time trend

    trim : float
        percentage of series at begin/end to exclude from break-period
        calculation in range [0, 0.333] (default=0.15)
    max_lags : int, optional
        The maximum number of lags to use when selecting lag length
    method : {"AIC", "BIC", "t-stat"}, optional
        The method to use when selecting the lag length

        - "AIC" - Select the minimum of the Akaike IC
        - "BIC" - Select the minimum of the Schwarz/Bayesian IC
        - "t-stat" - Select the minimum of the Schwarz/Bayesian IC

    Attributes
    ----------
    stationary_ : bool, whether the series in `fit` has a unit root
        (with structural break)
        more precisely, whether the null of the Zivot-Andrews test is rejected at
        ``p_threshold``
    test_statistic_ : float
        The ZA test statistic, of running ``ZivotAndrews`` on ``y`` in ``fit``
    pvalue_ : float : float
        p-value obtained when running ``ZivotAndrews`` on ``y`` in ``fit``
    usedlag_ : int
        The number of lags used in the test.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.stationarity import StationarityZivotAndrews
    >>>
    >>> X = load_airline()  # doctest: +SKIP
    >>> sty_est = StationarityZivotAndrews()  # doctest: +SKIP
    >>> sty_est.fit(X)  # doctest: +SKIP
    StationarityZivotAndrews(...)
    >>> sty_est.get_fitted_params()["stationary"]  # doctest: +SKIP
    False
    """

    _tags = {
        "X_inner_mtype": ["pd.Series", "nd.array"],
        "scitype:X": "Series",
        "python_dependencies": "arch",
    }

    def __init__(
        self,
        lags=None,
        trend="c",
        trim=0.15,
        max_lags=None,
        method="aic",
        p_threshold=0.05,
    ):
        self.lags = lags
        self.trend = trend
        self.trim = trim
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
        from arch.unitroot import ZivotAndrews

        p_threshold = self.p_threshold

        result = ZivotAndrews(
            y=X,
            lags=self.lags,
            trend=self.trend,
            trim=self.trim,
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
            special parameters are defined for a value, will return `"default"` set.
            There are no reserved values for parameter estimators.

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
            "lags": 5,
            "trend": "ct",
            "trim": 0.1,
            "max_lags": 10,
            "method": "t-stat",
            "p_threshold": 0.1,
        }

        return [params1, params2]


class StationarityVarianceRatio(BaseParamFitter):
    """Test for stationarity via the variance ratio test for random walks.

    Direct interface to ``VarianceRatio`` test from the `arch` package.

    Uses ``arch.unitroot.VarianceRatio`` as a test for unit roots,
    and derives a boolean statement whether a series is stationary.

    Also returns test results for the unit root test as fitted parameters.

    Parameters
    ----------
    lags : int
        The number of periods to used in the multi-period variance, which is
        the numerator of the test statistic.  Must be at least 2
    trend : {"n", "c"}, optional
        "c" allows for a non-zero drift in the random walk, while "n" requires
        that the increments to y are mean 0
    overlap : bool, optional
        Indicates whether to use all overlapping blocks.  Default is True.  If
        False, the number of observations in y minus 1 must be an exact
        multiple of lags.  If this condition is not satisfied, some values at
        the end of y will be discarded.
    robust : bool, optional
        Indicates whether to use heteroskedasticity robust inference. Default
        is True.
    debiased : bool, optional
        Indicates whether to use a debiased version of the test. Default is
        True. Only applicable if overlap is True.

    Attributes
    ----------
    stationary_ : bool
        whether the series in ``fit`` is stationary according to the test
        more precisely, whether the null of the variance ratio test is accepted at
        ``p_threshold``
    test_statistic_ : float
        The VR test statistic, of running ``VarianceRatio`` on ``y`` in ``fit``
    pvalue_ : float : float
        p-value obtained when running ``VarianceRatio`` on ``y`` in ``fit``
    usedlag_ : int
        The number of lags used in the test.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.stationarity import StationarityVarianceRatio
    >>>
    >>> X = load_airline()  # doctest: +SKIP
    >>> sty_est = StationarityVarianceRatio()  # doctest: +SKIP
    >>> sty_est.fit(X)  # doctest: +SKIP
    StationarityVarianceRatio(...)
    >>> sty_est.get_fitted_params()["stationary"]  # doctest: +SKIP
    True
    """

    _tags = {
        "X_inner_mtype": ["pd.Series", "nd.array"],
        "scitype:X": "Series",
        "python_dependencies": "arch",
    }

    def __init__(
        self,
        lags=2,
        trend="c",
        overlap=True,
        robust=True,
        debiased=True,
        p_threshold=0.05,
    ):
        self.lags = lags
        self.trend = trend
        self.overlap = overlap
        self.robust = robust
        self.debiased = debiased
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
        from arch.unitroot import VarianceRatio

        p_threshold = self.p_threshold

        result = VarianceRatio(
            y=X,
            lags=self.lags,
            trend=self.trend,
            overlap=self.overlap,
            robust=self.robust,
            debiased=self.debiased,
        )
        self.test_statistic_ = result.stat
        self.pvalue = result.pvalue
        self.stationary_ = result.pvalue > p_threshold
        self.used_lag_ = result._lags

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are no reserved values for parameter estimators.

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
            "lags": 5,
            "overlap": False,
            "robust": False,
            "debiased": False,
            "p_threshold": 0.1,
        }

        return [params1, params2]
