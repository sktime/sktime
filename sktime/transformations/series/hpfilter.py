# -*- coding: utf-8 -*-
"""
Implements Baxter-King bandpass filter transformation.

Please see the original library
(https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/filters/hp_filter.py)
"""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["klam-data", "pyyim", "mgorlin", "ken_maeda"]
__all__ = ["HPFilter"]


import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("statsmodels", severity="warning")


class HPFilter(BaseTransformer):
    """Filter a times series using the Hodrick-Prescott filter.
    This is a wrapper around statsmodels' hpfilter function
    (see 'sm.tsa.filters.bk_filter.hpfilter').

    Parameters
    ----------
    x : array_like
        The time series to filter, 1-d.
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
    >>> dta = sm.datasets.macrodata.load_pandas().data
    >>> index = pd.period_range('1959Q1', '2009Q3', freq='Q')
    >>> dta.set_index(index, inplace=True)
    >>> cycle, trend = sm.tsa.filters.hpfilter(dta.realgdp, 1600)
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
        super(HPFilter, self).__init__()

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
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        from statsmodels.tools.validation import PandasWrapper, array_like

        pw = PandasWrapper(X)
        X = array_like(X, 'x', ndim=1)
        nobs = len(X)
        I = sparse.eye(nobs, nobs)  # noqa:E741
        offsets = np.array([0, 1, 2])
        data = np.repeat([[1.], [-2.], [1.]], nobs, axis=1)
        K = sparse.dia_matrix((data, offsets), shape=(nobs - 2, nobs))

        use_umfpack = True
        trend = spsolve(I+self.lamb*K.T.dot(K), X, use_umfpack=use_umfpack)

        cycle = X - trend
        return pw.wrap(cycle, append='cycle')

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
        params = {"lamb": 1600}
        return params