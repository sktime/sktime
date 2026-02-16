# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimators for residuals."""

__author__ = ["HarshvirSandhu"]
__all__ = [
    "JarqueBera",
]

from sktime.param_est.base import BaseParamFitter


class JarqueBera(BaseParamFitter):
    """Jarque-Bera test of normality.

    Uses ``statsmodels.stats.stattools.jarque_bera`` as a test of normality.

    Parameters
    ----------
    axis : int, optional
        Axis to use if data has more than 1 dimension. Default is 0.

    Attributes
    ----------
    jb_stat_ : float
        The Jarque-Bera test statistic.
    pvalue_ : float
        The pvalue of the test statistic.
    skew_ : float
        Estimated skewness of the data.
    kurtosis_ : float
        Estimated kurtosis of the data.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.residuals import JarqueBera
    >>>
    >>> X = load_airline()
    >>> param_est = JarqueBera()
    >>> param_est.fit(X)
    JarqueBera(...)
    """

    _tags = {
        "authors": "HarshvirSandhu",
        "X_inner_mtype": "pd.Series",  # which types do _fit/_predict, support for X?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "python_dependencies": "statsmodels",
    }

    def __init__(self, axis=0):
        self.axis = axis
        super().__init__()

    def _fit(self, X):
        from statsmodels.stats.stattools import jarque_bera

        res = jarque_bera(resids=X, axis=self.axis)
        jb, jb_pv, skew, kurtosis = res
        self.jb_stat_ = jb
        self.pvalue_ = jb_pv
        self.skew_ = skew
        self.kurtosis_ = kurtosis

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
            "axis": 0,
        }

        return [params1, params2]
