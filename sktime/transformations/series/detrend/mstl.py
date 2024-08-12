# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements MSTL."""

__all__ = ["MSTL"]
__authors__ = ["luca-miniati"]

from collections.abc import Sequence
from typing import Optional, Union

import pandas as pd

from sktime.transformations.base import BaseTransformer


class MSTL(BaseTransformer):
    """Season-Trend decomposition using LOESS for multiple seasonalities.

    Direct interface for ``statsmodels.tsa.seasonal.MSTL`` for ``transform``,
    with ``sktime`` native extensions to allow use in forecasting pipelines.

    ``MSTL`` can be used to perform deseasonalization or decomposition:

    ``fit`` stores the decomposed values in self.trend_, self.seasonal_, and
    self.resid_.

    If ``return_components=False``, then ``transform`` returns a pd.Series of the
    deseasonalized values, i.e., trend plus residual component.
    The individual seasonal and residual components can be found in
    self.trend_ and self.resid_.

    If ``return_components=True``, then ``transform`` returns
    a full components decomposition,
    in a DataFrame with cols (for each input column),
    in this order:

    * "trend" - the trend component
    * "resid" - the residuals after de-trending, de-seasonalizing
    * "seasonal" - a single sum-of-seasonalities component, if
      ``periods`` is ``None``.
    * "seasonal_<period>" - the seasonal component(s),
      where <period> is an integer indicating the periodicity,
      one such component per element in ``periods``, if
      ``periods`` is an array-like of integers.

    ``MSTL`` performs ``inverse_transform`` by reconstituting the signal from its
    components, and can be used for pipelining in a ``TransformedTargetForecaster``,
    see examples below.

    * if ``periods`` are provided, the transformation will deseasonalize,
      and reseasonalize after forecast.
    * if ``periods`` are not provided, and ``return_components=False``,
      the forecast will be a pure trend forecast, using sum of trend and residual
      components.
    * if ``return_components=True``, the forecaster has access to
      all components, and can apply different forecasters to different components.

    See the examples below for usage.

    For automated detection of seasonalities using a custom seasonality detection
    algorithm, pipeline ``MSTL`` with the respective estimator, e.g.,
    ``SeasonalityACF``.

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
        decomposition. If None, no transform is applied. If ``auto``, a value will be
        estimated that maximizes the log-likelihood function.
    iterate : int, optional
        Number of iterations to use to refine the seasonal component.
    stl_kwargs : dict, optional
        Arguments to pass to STL.
    return_components : bool, default=False
        * if False, will return only the MSTL transformed series, same
          as trend plus residual component. The resulting series has the same
          number of columns as the input.

        * if True, will return all components of the decomposition,
            a multivariate series with DataFrame cols (for each input column):

            * "trend" - the trend component
            * "resid" - the residuals after de-trending, de-seasonalizing
            * "seasonal" - a single sum-of-seasonalities component, if
            ``periods`` is ``None``.
            * "seasonal_<period>" - the seasonal component(s),
              where <period> is an integer indicating the periodicity,
              one such component per element in ``periods``

            All components together sum up to the original series, in-sample.

    Attributes
    ----------
    trend_ : pd.Series
        Trend component of series seen in fit.
    resid_ : pd.Series
        Residuals component of series seen in fit.
    seasonal_ : pd.DataFrame
        If ``periods`` is ``None``, this contains a single column,
        with the sum of all seasonal components of the ``X`` seen in ``fit``.
        If ``periods`` is an array-like of integers, this consists
        of multiple columns ``seasonal_<period>``, each
        corresponding to a seasonal component of the series.

    References
    ----------
    [1] https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.MSTL.html

    Examples
    --------
    Simple use case: decompose a time series into trend, seasonal, residual components
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.series.detrend import MSTL
    >>> y = load_airline()
    >>> y.index = y.index.to_timestamp()
    >>> mstl = MSTL(return_components=True)
    >>> mstl.fit(y)
    >>> res = mstl.transform(y)
    >>> res.plot()  # doctest: +SKIP
    >>> plt.tight_layout()  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    MSTL can be pipelined with a forecaster for multiple deseasonalized forecasts.
    The following example uses a simple trend forecaster, applied
    to a series deseasonalized with MSTL at periods 2 and 12.
    After the trend forecast, the seasonal components
    are added back to the forecast automatically.
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.series.detrend import MSTL
    >>> from sktime.forecasting.trend import TrendForecaster
    >>>
    >>> mstl_trafo = MSTL(periods=[2, 12])
    >>> mstl_deseason_fcst = mstl_trafo * TrendForecaster()
    >>> y = load_airline()
    >>> mstl_deseason_fcst.fit(y, fh=[1, 2, 3])
    >>> y_pred = mstl_deseason_fcst.predict()

    MSTL can also be used to make forecasts using the full component decomposition.
    For this, set ``return_components=True`` when pipelining.
    The forecaster in the pipeline will then be given a multivariate series
    with the components as columns,
    i.e., "trend", "resid", "seasonal_2", "seasonal_12".
    To apply different forecasters to different components, use a
    ``ColumnEnsembleForecaster``; to apply the same forecaster to all components,
    simply pipeline with the forecaster.
    The following example uses a ``TrendForecaster`` for the trend,
    a seasonal naive forecaster for the seasonal components, with different
    seasonalities, and a naive forecaster for the residuals.
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.series.detrend import MSTL
    >>> from sktime.forecasting.compose import ColumnEnsembleForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import TrendForecaster
    >>>
    >>> mstl_trafo_comp = MSTL(periods=[2, 12], return_components=True)
    >>> mstl_component_fcst = mstl_trafo_comp * ColumnEnsembleForecaster(
    ...     [
    ...         ("trend", TrendForecaster(), "trend"),
    ...         ("sp2", NaiveForecaster(strategy="last", sp=2), "seasonal_2"),
    ...         ("sp12", NaiveForecaster(strategy="last", sp=12), "seasonal_12"),
    ...         ("residual", NaiveForecaster(strategy="last"), "resid"),
    ...     ]
    ... )
    >>> y = load_airline()
    >>> mstl_component_fcst.fit(y, fh=[1, 2, 3])
    >>> y_pred = mstl_component_fcst.predict()
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["luca-miniati", "fkiraly"],
        "maintainers": ["luca-miniati"],
        "python_dependencies": "statsmodels",
        # estimator type
        # --------------
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
    }

    def __init__(
        self,
        *,
        periods: Optional[Union[int, Sequence[int]]] = None,
        windows: Optional[Union[int, Sequence[int]]] = None,
        lmbda: Optional[Union[float, str]] = None,
        iterate: Optional[int] = 2,
        stl_kwargs: Optional[dict[str, Union[int, bool, None]]] = None,
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

        self.seasonal_ = _coerce_to_df(self.mstl_.seasonal)
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
        if self.return_components or self.periods is None:
            if isinstance(X, pd.DataFrame):
                row_sums = X.sum(axis=1)
            else:
                row_sums = X
            row_sums.name = self._X.name
            return row_sums
        # otherwise, we make naive seasonal forecasts, and add them to "transformed"
        # since "transformed" is trend + resid, this will restore the full series
        from sktime.forecasting.base import ForecastingHorizon
        from sktime.forecasting.naive import NaiveForecaster

        seasonal = self.seasonal_

        fcsts = []
        for period in self.periods:
            nf = NaiveForecaster(strategy="last", sp=period)
            fh = ForecastingHorizon(X.index, is_relative=False)
            sp_ix = f"seasonal_{period}"
            nf_pred = nf.fit(seasonal[sp_ix], fh=fh).predict()
            fcsts.append(nf_pred)
        fcsts = pd.DataFrame(fcsts).T
        return X + fcsts.sum(axis=1)

    def _make_return_object(self, X, mstl):
        seasonal = _coerce_to_df(mstl.seasonal)
        seasonal_sum = seasonal.sum(axis=1)
        # deseasonalize only
        transformed = pd.Series(X.values - seasonal_sum, index=X.index)
        # transformed = pd.Series(X.values - stl.seasonal - stl.trend, index=X.index)

        if self.return_components:
            resid = pd.Series(mstl.resid, index=X.index)
            trend = pd.Series(mstl.trend, index=X.index)

            ret = {"trend": trend, "resid": resid}

            for column_name, column_data in seasonal.items():
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
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict , default = {}
            parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in `params
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
        params3 = {
            "periods": [3, 12],
            "return_components": True,
        }

        return [params1, params2, params3]


def _coerce_to_df(x):
    """Coerce pd.Series or pd.DataFrame to pd.DataFrame."""
    if not isinstance(x, pd.DataFrame):
        if isinstance(x, pd.Series):
            x = pd.DataFrame(x)
        else:
            raise ValueError(f"Unexpected input type {type(x)}")
    return x
