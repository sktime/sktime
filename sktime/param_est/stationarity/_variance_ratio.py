# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimator for stationarity via variance ratio test from arch package."""

__author__ = ["Vasudeva-bit"]
__all__ = ["StationarityVarianceRatio"]

from sktime.param_est.base import BaseParamFitter


class StationarityVarianceRatio(BaseParamFitter):
    """Test for stationarity via the variance ratio test for random walks.

    Direct interface to ``VarianceRatio`` test from the ``arch`` package.

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
            "overlap": False,
            "robust": False,
            "debiased": False,
            "p_threshold": 0.1,
        }

        return [params1, params2]
