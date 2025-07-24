# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimators for autocorrelation."""

__author__ = ["HarshvirSandhu"]
__all__ = [
    "AcorrLjungbox",
]

from sktime.param_est.base import BaseParamFitter


class AcorrLjungbox(BaseParamFitter):
    """Performs the Ljung-Box test of autocorrelation in residuals.

    Uses ``statsmodels.stats.diagnostic.acorr_ljungbox`` as a test of
    autocorrelation in residuals.

    Parameters
    ----------
    lags : int or array_like, default=None
        If lags is an integer then this is taken to be the largest lag that is included.
        If lags is an array, then all lags are included upto largest lag in the list.
        If lags is None, then the default maxlag is min(10, nobs // 5).
            The default number of lags changes if period is set.

    boxpierce : bool, default=False
        If true, then Box-Pierce test results are also returned.


    Attributes
    ----------
    lb_statistic_ : array
        The Ljung-Box test statistic.
    lb_pvalue_ : array
        The p-value based on chi-square distribution. The p-value is computed as:
          1 - chi2.cdf(lb_stat, dof) where dof is lag - model_df.
        If lag - model_df <= 0, then NaN is returned for the pvalue.
    bp_stat_ : array
        The Box-Pierce test statistic.
    bp_pvalue_ : array
        The p-value based for Box-Pierce test on chi-square distribution, computed as:
         1 - chi2.cdf(bp_stat, dof) where dof is lag - model_df.
        If lag - model_df <= 0, then NaN is returned for the pvalue.
    lags_ : int
        The truncation lag parameter.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.stationarity import AcorrLjungbox
    >>>
    >>> X = load_airline()  # doctest: +SKIP
    >>> sty_est = AcorrLjungbox()  # doctest: +SKIP
    >>> sty_est.fit(X)  # doctest: +SKIP
    AcorrLjungbox(...)
    """

    _tags = {
        "authors": "HarshvirSandhu",
        "X_inner_mtype": "pd.Series",  # which types do _fit/_predict, support for X?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "python_dependencies": "statsmodels",
    }

    def __init__(self, lags=1, boxpierce=False):
        self.lags = lags
        self.boxpierce = boxpierce
        super().__init__()

    def _fit(self, X):
        from statsmodels.stats.diagnostic import acorr_ljungbox

        res = acorr_ljungbox(x=X, lags=self.lags, boxpierce=self.boxpierce)
        self.lb_statistic_ = res["lb_stat"].to_numpy()
        self.lb_pvalue_ = res["lb_pvalue"].to_numpy()
        if self.boxpierce:
            self.bp_statistic_ = res["bp_stat"].to_numpy()
            self.bp_pvalue_ = res["bp_pvalue"].to_numpy()
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
        params2 = {
            "lags": 1,
            "boxpierce": True,
        }

        return [params1, params2]
