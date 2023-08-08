# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements MSTL."""

__all__ = ["MSTL"]
__authors__ = ["luca-miniati"]

from typing import Optional

import pandas as pd

from sktime.transformations.base import BaseTransformer


class MSTL(BaseTransformer):
    """Season-Trend decomposition using LOESS for multiple seasonalities.

    Direct interface for `statsmodels.tsa.seasonal.MSTL`.

    Parameters
    ----------
    endog : array_like
        Data to be decomposed. Must be squeezable to 1-d.
    periods : {int, array_like, None}, optional
        Periodicity of the seasonal components. If None and endog is a pandas Series or
        DataFrame, attempts to determine from endog. If endog is a ndarray, periods
        must be provided.
    windows : {int, array_like, None}, optional
        Length of the seasonal smoothers for each corresponding period. Must be an odd
        integer, and should normally be >= 7 (default). If None then default values
        determined using 7 + 4 * np.arange(1, n + 1, 1) where n is number of seasonal
        components.
    lmbda : {float, str, None}, optional
        The lambda parameter for the Box-Cox transform to be applied to endog prior to
        decomposition. If None, no transform is applied. If “auto”, a value will be
        estimated that maximizes the log-likelihood function.
    iterate : int, optional
        Number of iterations to use to refine the seasonal component.
    stl_kwargs : dict, optional
        Arguments to pass to STL.

    References
    ----------
    [1] https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.MSTL.html

    Examples
    --------
    >>> from sktime.forecasting.mstl import MSTL  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> y = load_airline()
    >>> mstl = MSTL()  # doctest: +SKIP
    >>> res = mstl.fit(y)  # doctest: +SKIP
    >>> res.plot()  # doctest: +SKIP
    >>> plt.tight_layout()  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP
    """

    _tags = {
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        periods=None,
        windows=None,
        lmbda=None,
        iterate: Optional[int] = 2,
        stl_kwargs: Optional[dict] = None,
        return_components=False,
    ):
        self.periods = periods
        self.windows = windows
        self.lmbda = lmbda
        self.iterate = iterate
        self.stl_kwargs = stl_kwargs
        self.return_components = return_components

        super().__init__()

    def _fit(self, X, y=None):
        from statsmodels.tsa.seasonal import MSTL as _MSTL

        self.mstl_ = _MSTL(
            y,
            periods=self.periods,
            windows=self.windows,
            lmbda=self.lmbda,
            iterate=self.iterate,
            stl_kwargs=self.stl_kwargs,
        ).fit()

        self.seasonal_ = pd.Series(self.mstl_.seasonal, index=X.index)
        self.resid_ = pd.Series(self.mstl_.resid, index=X.index)
        self.trend_ = pd.Series(self.mstl_.trend, index=X.index)

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series, Panel, or Hierarchical data, of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series, Panel, or Hierarchical data, of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        from statsmodels.tsa.seasonal import MSTL as _MSTL

        # fit again if indices not seen, but don't store anything
        if not X.index.equals(self._X.index):
            X_full = X.combine_first(self._X)
            new_mstl = _MSTL(
                X_full.values,
                periods=self.periods,
                windows=self.windows,
                lmbda=self.lmbda,
                iterate=self.iterate,
                stl_kwargs=self.stl_kwargs,
            ).fit()

            ret_obj = self._make_return_object(X_full, new_mstl)
        else:
            ret_obj = self._make_return_object(X, self.mstl_)

        return ret_obj

    def _make_return_object(self, X, mstl):
        # deseasonalize only
        transformed = pd.Series(X.values - mstl.seasonal, index=X.index)
        # transformed = pd.Series(X.values - stl.seasonal - stl.trend, index=X.index)

        if self.return_components:
            seasonal = pd.Series(mstl.seasonal, index=X.index)
            resid = pd.Series(mstl.resid, index=X.index)
            trend = pd.Series(mstl.trend, index=X.index)

            ret = pd.DataFrame(
                {
                    "transformed": transformed,
                    "seasonal": seasonal,
                    "trend": trend,
                    "resid": resid,
                }
            )
        else:
            ret = transformed

        return ret

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str , default = "default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict , default = {}
            arameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params
        """
        params1 = {}
        params2 = {
            "periods": [1, 12],
            "windows": 9,
            "lmbda": "auto",
            "iterate": 10,
            "stl_kwargs": {"trend_deg": 0},
        }

        return [params1, params2]
