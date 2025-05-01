"""Interface to Baxter-King bandpass filter from ``statsmodels``.

Interfaces ``bk_filter`` from ``statsmodels.tsa.filters``.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["klam-data", "pyyim", "mgorlin"]
__all__ = ["BKFilter"]


import pandas as pd

from sktime.transformations.base import BaseTransformer


class BKFilter(BaseTransformer):
    """Filter a times series using the Baxter-King filter.

    This is a wrapper around the ``bkfilter`` function from ``statsmodels``.
    (see ``statsmodels.tsa.filters.bk_filter.bkfilter``).

    The Baxter-King filter is intended for economic and econometric time series
    data and deals with the periodicity of the business cycle. Applying their
    band-pass filter to a series will produce a new series that does not contain
    fluctuations at a higher or lower frequency than those of the business cycle.
    Baxter-King follow Burns and Mitchell's work on business cycles, which suggests
    that U.S. business cycles typically last from 1.5 to 8 years.

    Parameters
    ----------
    low : float
        Minimum period for oscillations. Baxter and King recommend a value of 6
        for quarterly data and 1.5 for annual data.

    high : float
        Maximum period for oscillations. BK recommend 32 for U.S. business cycle
        quarterly data and 8 for annual data.

    K : int
        Lead-lag length of the filter. Baxter and King suggest a truncation
        length of 12 for quarterly data and 3 for annual data.

    Notes
    -----
    Returns a centered weighted moving average of the original series.

    References
    ----------
    Baxter, M. and R. G. King. "Measuring Business Cycles: Approximate
        Band-Pass Filters for Economic Time Series." *Review of Economics and
        Statistics*, 1999, 81(4), 575-593.

    Examples
    --------
    >>> from sktime.transformations.series.bkfilter import BKFilter # doctest: +SKIP
    >>> import pandas as pd # doctest: +SKIP
    >>> import statsmodels.api as sm # doctest: +SKIP
    >>> dta = sm.datasets.macrodata.load_pandas().data # doctest: +SKIP
    >>> index = pd.date_range(start='1959Q1', end='2009Q4', freq='Q') # doctest: +SKIP
    >>> dta.set_index(index, inplace=True) # doctest: +SKIP
    >>> bk = BKFilter(6, 24, 12) # doctest: +SKIP
    >>> cycles = bk.fit_transform(X=dta[['realinv']]) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["klam-data", "pyyim", "mgorlin"],
        "maintainers": "klam-data",
        "python_dependencies": "statsmodels",
        # estimator type
        # --------------
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
        "capability:missing_values": False,  # can estimator handle missing data?
        "remember_data": False,  # whether all data seen is remembered as self._X
    }

    def __init__(
        self,
        low=6,
        high=32,
        K=12,
    ):
        self.low = low
        self.high = high
        self.K = K
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
        from statsmodels.tsa.filters.bk_filter import bkfilter

        kwargs = {"low": self.low, "high": self.high, "K": self.K}
        return bkfilter(X, **kwargs)

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
        params1 = {"low": 5, "high": 24, "K": 4}
        params2 = {}
        return [params1, params2]
