"""Interface to Hodrick-Prescott filter from `statsmodels`.

Please see the original library
(https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/filters/hp_filter.py)
Interfaces `hp_filter` from `statsmodels.tsa.filters`.
"""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ken_maeda"]
__all__ = ["HPFilter"]


import pandas as pd

from sktime.transformations.base import BaseTransformer


class HPFilter(BaseTransformer):
    """Filter a times series using the Hodrick-Prescott filter.

    This is a wrapper around the `hpfilter` function from `statsmodels`.
    (see `statsmodels.tsa.filters.hp_filter.hpfilter`).

    Parameters
    ----------
    lamb : float
        The Hodrick-Prescott smoothing parameter. A value of 1600 is
        suggested for quarterly data. Ravn and Uhlig suggest using a value
        of 6.25 (1600/4**4) for annual data and 129600 (1600*3**4) for monthly
        data.

    Notes
    -----
    The HP filter removes a smooth trend
    ----------
    Hodrick, R.J, and E. C. Prescott. 1980. "Postwar U.S. Business Cycles: An
        Empirical Investigation." `Carnegie Mellon University discussion
        paper no. 451`.
    Ravn, M.O and H. Uhlig. 2002. "Notes On Adjusted the Hodrick-Prescott
        Filter for the Frequency of Observations." `The Review of Economics and
        Statistics`, 84(2), 371-80.

    Examples
    --------
    >>> from sktime.transformations.series.hpfilter import HPFilter # doctest: +SKIP
    >>> import pandas as pd # doctest: +SKIP
    >>> import statsmodels.api as sm # doctest: +SKIP
    >>> dta = sm.datasets.macrodata.load_pandas().data # doctest: +SKIP
    >>> index = pd.period_range('1959Q1', '2009Q3', freq='Q') # doctest: +SKIP
    >>> dta.set_index(index, inplace=True) # doctest: +SKIP
    >>> hp = HPFilter(1600) # doctest: +SKIP
    >>> cycles = hp.fit_transform(X=dta[['realinv']]) # doctest: +SKIP
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": [
            pd.RangeIndex
        ],  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "capability:unequal_length": True,
        # can the transformer handle unequal length time series (if passed Panel)?
        "handles-missing-data": False,  # can estimator handle missing data?
        "remember_data": False,  # whether all data seen is remembered as self._X
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        lamb=1600,
    ):
        self.lamb = lamb
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : array_like, A 1d array

        Returns
        -------
        transformed cyclical version of X
        """
        from statsmodels.tsa.filters.hp_filter import hpfilter

        kwargs = {"lamb": self.lamb}
        return hpfilter(X, **kwargs)[0]

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
        params1 = {"lamb": 1600}
        params2 = {}
        return [params1, params2]
