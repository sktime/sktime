"""Interface to Christiano Fitzgerald asymmetric, random walk filter from `statsmodels`.

Interfaces `cf_filter` from `statsmodels.tsa.filters`.
"""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ken-maeda"]
__all__ = ["CFFilter"]


import pandas as pd

from sktime.transformations.base import BaseTransformer


class CFFilter(BaseTransformer):
    """Filter a times series using the Christiano Fitzgerald filter.

    This is a wrapper around the `cffilter` function from `statsmodels`.
    (see `statsmodels.tsa.filters.cf_filter.cffilter`).

    Parameters
    ----------
    low : float, optional, default = 6.0
        Minimum period of oscillations. Features below low periodicity
        are filtered out. For quarterly data, the default of 6 gives
        1.5 years periodicity.

    high : float, optional, default = 32.0
        Maximum period of oscillations. Features above high periodicity
        are filtered out. For quarterly data, the default of 32 gives
        8 year periodicity.

    drift : bool, optional, default = True
        Whether or not to substract a trend from the data.
        The trend is estimated as np.arange(nobs)*(x[-1] -x[0])/(len(x)-1).
        > X : argument of CFFilter._transform()
        > x : If X is 1d, X=x. If 2d, x is assumed to be in columns.
        > nobs : len(x)

    Examples
    --------
    >>> from sktime.transformations.series.cffilter import CFFilter # doctest: +SKIP
    >>> import pandas as pd # doctest: +SKIP
    >>> import statsmodels.api as sm # doctest: +SKIP
    >>> dta = sm.datasets.macrodata.load_pandas().data # doctest: +SKIP
    >>> index = pd.date_range(start='1959Q1', end='2009Q4', freq='Q') # doctest: +SKIP
    >>> dta.set_index(index, inplace=True) # doctest: +SKIP
    >>> cf = CFFilter(6, 24, True) # doctest: +SKIP
    >>> cycles = cf.fit_transform(X=dta[['realinv']]) # doctest: +SKIP
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
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
        low=6,
        high=32,
        drift=True,
    ):
        self.low = low
        self.high = high
        self.drift = drift
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : array_like
        A 1 or 2d ndarray. If 2d, variables are assumed to be in columns.

        Returns
        -------
        transformed cyclical version of X
        """
        from statsmodels.tsa.filters.cf_filter import cffilter

        kwargs = {"low": self.low, "high": self.high, "drift": self.drift}
        return cffilter(X, **kwargs)[0]

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
        params1 = {"low": 8, "high": 26, "drift": False}
        params2 = {}
        return [params1, params2]
