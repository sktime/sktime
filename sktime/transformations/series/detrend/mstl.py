# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements MSTL."""

__all__ = ["MSTL"]
__authors__ = ["luca-miniati"]

from typing import Dict, Optional, Sequence, Union

import pandas as pd

from sktime.transformations.base import BaseTransformer


class MSTL(BaseTransformer):
    """Season-Trend decomposition using LOESS for multiple seasonalities.

    Direct interface for ``statsmodels.tsa.seasonal.MSTL``.
    ``MSTL`` can be used to perform deseasonalization or decomposition:

    `fit` stores the decomposed values in self.trend_, self.seasonal_, and self.resid_.
    If `return_components=False`, then `transform` returns a pd.Series of the
    deseasonalized values. The seasonal and residual components can be found in
    self.trend_ and self.resid_.
    If `return_components=True`, then `transform` returns the transformed series, as
    well as three components as variables in the returned multivariate series
    (DataFrame cols)
        "transformed" - the transformed series
        "seasonal" - the seasonal component(s), summed up if multiple
        "trend" - the trend component
        "resid" - the residuals after de-trending, de-seasonalizing

    ``MSTL`` performs ``inverse_transform`` by summing any components,
    and can be used for pipelining in a ``TransformedTargetForecaster``.

    Important: for separate forecasts of trend and seasonalities, and an
    inverse transform that respects seasonality, ensure
    that ``return_components=True`` is set, otherwise the inverse will just
    return the trend component.

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
        decomposition. If None, no transform is applied. If `auto`, a value will be
        estimated that maximizes the log-likelihood function.
    iterate : int, optional
        Number of iterations to use to refine the seasonal component.
    stl_kwargs : dict, optional
        Arguments to pass to STL.
    return_components : bool, default=False
        if False, will return only the MSTL transformed series
        if True, will return the transformed series, as well as three components
            as variables in the returned multivariate series (DataFrame cols)
            "transformed" - the transformed series
            "seasonal" - the seasonal component(s), summed up if multiple
            "trend" - the trend component
            "resid" - the residuals after de-trending, de-seasonalizing

    Attributes
    ----------
    trend_ : pd.Series
        Trend component of series seen in fit.
    seasonal_ : pd.Series or list of pd.Series
        If `periods` is a single value, this contains the seasonal component of the
        series observed during fitting.
        If `periods` is a list of values, this can contain multiple pd.Series, each
        corresponding to a different period.
    resid_ : pd.Series
        Residuals component of series seen in fit.

    References
    ----------
    [1] https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.MSTL.html

    Examples
    --------
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.series.detrend.mstl import MSTL
    >>> y = load_airline()
    >>> y.index = y.index.to_timestamp()
    >>> mstl = MSTL(return_components=True)  # doctest: +SKIP
    >>> fitted = mstl.fit(y)  # doctest: +SKIP
    >>> res = fitted.transform(y)  # doctest: +SKIP
    >>> res.plot()  # doctest: +SKIP
    >>> plt.tight_layout()  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "None",
        "transform-returns-same-time-index": True,
        "univariate-only": True,
        "capability:inverse_transform": True,
        "capability:inverse_transform:exact": False,
        "skip-inverse-transform": False,
        "fit_is_empty": False,
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        *,
        periods: Optional[Union[int, Sequence[int]]] = None,
        windows: Optional[Union[int, Sequence[int]]] = None,
        lmbda: Optional[Union[float, str]] = None,
        iterate: Optional[int] = 2,
        stl_kwargs: Optional[Dict[str, Union[int, bool, None]]] = None,
        return_components: bool = False,
    ):
        self.periods = periods
        self.windows = windows
        self.lmbda = lmbda
        self.iterate = iterate
        self.stl_kwargs = stl_kwargs
        self.return_components = return_components
        self._X = None

        super().__init__()

    def _fit(self, X, y=None):
        from statsmodels.tsa.seasonal import MSTL as _MSTL

        self._X = X

        self.mstl_ = _MSTL(
            X,
            periods=self.periods,
            windows=self.windows,
            lmbda=self.lmbda,
            iterate=self.iterate,
            stl_kwargs=self.stl_kwargs,
        ).fit()

        self.seasonal_ = self.mstl_.seasonal
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

    def _inverse_transform(self, X, y=None):
        # for inverse transform, we sum up the columns
        # this will be close if return_components=True
        row_sums = X.sum(axis=1)
        row_sums.columns = self.fit_column_names
        return row_sums

    def _make_return_object(self, X, mstl):
        if len(mstl.seasonal.shape) > 1:
            seasonal = mstl.seasonal.sum(axis=1)
        else:
            seasonal = mstl.seasonal
        # deseasonalize only
        transformed = pd.Series(X.values - seasonal, index=X.index)
        # transformed = pd.Series(X.values - stl.seasonal - stl.trend, index=X.index)

        if self.return_components:
            resid = pd.Series(mstl.resid, index=X.index)
            trend = pd.Series(mstl.trend, index=X.index)

            ret = {
                "transformed": transformed,
                "trend": trend,
                "resid": resid,
            }

            for column_name, column_data in mstl.seasonal.iteritems():
                ret[column_name] = column_data

            ret = pd.DataFrame(ret)
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
            parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params
        """
        params1 = {
            "periods": 3,
            "windows": 3,
        }
        params2 = {
            "periods": [3, 24],
            "windows": [3, 9],
            "lmbda": "auto",
            "iterate": 10,
            "stl_kwargs": {"trend_deg": 0},
        }

        return [params1, params2]
