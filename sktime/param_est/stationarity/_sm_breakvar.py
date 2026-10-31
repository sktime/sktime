# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimators for stationarity."""

from sktime.param_est.base import BaseParamFitter


class BreakvarHeteroskedasticityTest(BaseParamFitter):
    """Variance break test for heteroskedasticity.

    Uses ``statsmodels.tsa.stattools.breakvar_heteroskedasticity_test`` to test whether
    the sum-of-squares in the first subset of the sample is significantly different
    than the sum-of-squares in the last subset of the sample.

    The null hypothesis is of no heteroskedasticity.

    In literature, this test is typically applied to residuals of a fitted model,
    but can be applied to any time series.

    It can also be used as a weak test for non-stationarity,
    as hetoroskedasticity implies non-stationarity, but not vice versa.

    Parameters
    ----------
    p_threshold : float, optional, default=0.05
        significance threshold to apply in testing for heteroskedasticity.

    subset_length : float, default=1/3
        Length of the subsets to test.

    alternative : str, 'increasing', 'decreasing' or 'two-sided', default='two-sided'
        This specifies the alternative for the p-value calculation.

    use_f : bool, optional
        Whether or not to compare against the asymptotic distribution (chi-squared)
        or the approximate small-sample distribution (F).
        Default is True (i.e. default is to compare against an F distribution).

    Attributes
    ----------
    stationary_ : bool
        whether the series in ``fit`` is homoskedastic according to the test,
        more precisely, whether the null hypothesis is rejected at ``p_threshold``.
        Homoskedasticity is implied by stationarity, but not vice versa,
        therefore False implies non-stationarity, but True does not imply stationarity.
    bh_statistic_ : float
        Test statistic(s) H(h).
    pvalue_ : float
        p-value(s) of test statistic(s).

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.stationarity import BreakvarHeteroskedasticityTest
    >>>
    >>> X = load_airline()
    >>> sty_est = BreakvarHeteroskedasticityTest()
    >>> sty_est.fit(X)
    BreakvarHeteroskedasticityTest(...)
    """

    _tags = {
        "authors": "HarshvirSandhu",
        "X_inner_mtype": "pd.Series",  # which types do _fit/_predict, support for X?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        p_threshold=0.05,
        subset_length=1 / 3,
        alternative="two-sided",
        use_f=True,
    ):
        self.subset_length = subset_length
        self.alternative = alternative
        self.use_f = use_f
        self.p_threshold = p_threshold

        super().__init__()

    def _fit(self, X):
        from statsmodels.tsa.stattools import breakvar_heteroskedasticity_test

        res = breakvar_heteroskedasticity_test(
            resid=X,
            subset_length=self.subset_length,
            alternative=self.alternative,
            use_f=self.use_f,
        )
        test_statistic, p_value = res
        self.bh_statistic_ = test_statistic
        self.pvalue_ = p_value

        self.stationary_ = p_value > self.p_threshold

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

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
        params2 = {"subset_length": 1 / 2, "alternative": "decreasing"}

        return [params1, params2]
