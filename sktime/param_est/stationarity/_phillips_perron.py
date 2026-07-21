# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimator for stationarity via Phillips-Perron test from arch package."""

__author__ = ["Vasudeva-bit"]
__all__ = ["StationarityPhillipsPerron"]

from sktime.param_est.base import BaseParamFitter


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
            "test_type": "rho",
            "p_threshold": 0.1,
        }

        return [params1, params2]
