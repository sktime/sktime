# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements back-adapters for skforecast reduction models."""

from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

__author__ = ["Abhay-Lejith", "yarnabrina"]


class SkforecastAutoreg(BaseForecaster):
    """Adapter for ``skforecast.ForecasterAutoreg.ForecasterAutoreg`` class [1]_.

    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API
    lags : int, list, numpy ndarray, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.

            - ``int``: include lags from 1 to ``lags`` (included).
            - ``list``, ``1d numpy ndarray`` or ``range``: include only lags present in
            ``lags``, all elements must be int.

    transformer_y : object transformer (preprocessor), default ``None``
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: ``fit``, ``transform``, ``fit_transform`` and
        ``inverse_transform``. ``ColumnTransformers`` are not allowed since they do not
        have ``inverse_transform`` method. The transformation is applied to ``y`` before
        training the forecaster.
    transformer_exog : object transformer (preprocessor), default ``None``
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to ``exog`` before training the
        forecaster. ``inverse_transform`` is not available when using
        ``ColumnTransformers``.
    weight_func : Callable, default ``None``
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if ``regressor`` does not have the argument ``sample_weight`` in its
        ``fit`` method. The resulting ``sample_weight`` cannot have negative values.
    differentiation : int, default ``None``
        Order of differencing applied to the time series before training the forecaster.
        If ``None``, no differencing is applied. The order of differentiation is the
        number of times the differencing operation is applied to a time series.
        Differencing involves computing the differences between consecutive data points
        in the series. Differentiation is reversed in the output of ``predict()`` and
        ``predict_interval()``.
    fit_kwargs : dict, default ``None``
        Additional arguments to be passed to the ``fit`` method of the regressor.
    binner_kwargs : dict, default `None`
        Additional arguments to pass to the `KBinsDiscretizer` used to discretize the
        residuals into k bins according to the predicted values associated with each
        residual. The `encode' argument is always set to 'ordinal' and `dtype' to
        np.float64.

    References
    ----------
    .. [1]
        https://skforecast.org/latest/api/forecasterautoreg#forecasterautoreg

    Examples
    --------
    >>> from sktime.forecasting.compose import SkforecastAutoreg

    Without exogenous features

    >>> from sklearn.linear_model import LinearRegression
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = SkforecastAutoreg(  # doctest: +SKIP
    ...     LinearRegression(), 2
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    SkforecastAutoreg(lags=2, regressor=LinearRegression())
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    >>> y_pred_int = forecaster.predict_interval(  # doctest: +SKIP
    ...     fh=[2], coverage=[0.9, 0.95]
    ... )
    >>> y_pred_qtl = forecaster.predict_quantiles(  # doctest: +SKIP
    ...     fh=[1, 3], alpha=[0.8, 0.3, 0.2, 0.7]
    ... )

    With exogenous features

    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sktime.datasets import load_longley
    >>> y, X = load_longley()
    >>> y_train = y.head(n=12)
    >>> y_test = y.tail(n=4)
    >>> X_train = X.head(n=12)
    >>> X_test = X.tail(n=4)
    >>> forecaster = SkforecastAutoreg(  # doctest: +SKIP
    ...     RandomForestRegressor(), [2, 4]
    ... )
    >>> forecaster.fit(y_train, X=X_train)  # doctest: +SKIP
    SkforecastAutoreg(lags=[2, 4], regressor=RandomForestRegressor())
    >>> y_pred = forecaster.predict(fh=[1, 2, 3], X=X_test)  # doctest: +SKIP
    >>> y_pred_int = forecaster.predict_interval(  # doctest: +SKIP
    ...     fh=[1, 3], X=X_test, coverage=[0.6, 0.4]
    ... )
    >>> y_pred_qtl = forecaster.predict_quantiles(  # doctest: +SKIP
    ...     fh=[1, 3], X=X_test, alpha=[0.01, 0.5]
    ... )

    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "JoaquinAmatRodrigo",
            "JavierEscobarOrtiz",
            "FernandoCarazoMelo",
            "fernando-carazo",
            "Abhay-Lejith",
            "yarnabrina",
        ],
        # JoaquinAmatRodrigo, JavierEscobarOrtiz, FernandoCarazoMelo for skforecast
        "maintainers": ["Abhay-Lejith", "yarnabrina"],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:categorical_in_X": True,
        "python_version": ">=3.8,<3.13",
        "python_dependencies": ["skforecast>=0.12.1"],
    }

    def __init__(
        self: "SkforecastAutoreg",
        regressor: object,
        lags: Union[int, np.ndarray, list],
        transformer_y: Optional[object] = None,
        transformer_exog: Optional[object] = None,
        weight_func: Optional[Callable] = None,
        differentiation: Optional[int] = None,
        fit_kwargs: Optional[dict] = None,
        binner_kwargs: Optional[dict] = None,
    ) -> None:
        self.regressor = regressor
        self.lags = lags
        self.transformer_y = transformer_y
        self.transformer_exog = transformer_exog
        self.weight_func = weight_func
        self.differentiation = differentiation
        self.fit_kwargs = fit_kwargs
        self.binner_kwargs = binner_kwargs

        super().__init__()

        self._regressor = None
        self._forecaster = None
        self._transformer_y = None
        self._transformer_exog = None

        self._clone_estimators()

    def _clone_estimators(self: "SkforecastAutoreg"):
        """Clone the regressor and transformers."""
        from sklearn.base import clone

        self._regressor = clone(self.regressor)

        if self.transformer_y:
            self._transformer_y = clone(self.transformer_y)

        if self.transformer_exog:
            self._transformer_exog = clone(self.transformer_exog)

    def _create_forecaster(self: "SkforecastAutoreg"):
        """Create ``skforecast.ForecasterAutoreg.ForecasterAutoreg`` model."""
        from skforecast.ForecasterAutoreg import ForecasterAutoreg

        return ForecasterAutoreg(
            self._regressor,
            self.lags,
            transformer_y=self._transformer_y,
            transformer_exog=self._transformer_exog,
            weight_func=self.weight_func,
            differentiation=self.differentiation,
            fit_kwargs=self.fit_kwargs,
            binner_kwargs=self.binner_kwargs,
        )

    @staticmethod
    def _coerce_column_names(X: Optional[pd.DataFrame]):
        if X is None:
            return None

        return X.rename(columns=lambda column_name: str(column_name))

    @staticmethod
    def _coerce_int_to_range_index(df):
        new_df = df.copy(deep=True)
        start = new_df.index[0]
        stop = new_df.index[-1] + 1
        if len(new_df.index) == 1:
            step = 1
        else:
            step = new_df.index[1] - start

        new_index = pd.RangeIndex(start, stop, step)
        # testing if RangeIndex is matching the original integer Index.
        # this will fail if indices are not equally spaced apart.
        # exception is caught in _make_index_compatible.
        np.testing.assert_array_equal(new_df.index, new_index)
        # assigning RangeIndex to the new DataFrame
        new_df.index = new_index
        return new_df

    @staticmethod
    def _coerce_period_to_datetime_index(df):
        new_df = df.copy(deep=True)
        period_freq = new_df.index.freq
        # converting the period index to the timestamp at the 'end' of the period
        # Example:        Period                   Datetime
        #          '2024-01-12 00:00' --> '2024-01-12 00:59:59.999999999'
        #          '2024-01-12 01:00' --> '2024-01-12 01:59:59.999999999'
        new_df.index = new_df.index.to_timestamp(how="e")
        new_df.index.freq = period_freq
        return new_df

    def _make_index_compatible(self, df, input_var):
        if df is None:
            return None

        if isinstance(df.index, (pd.RangeIndex, pd.DatetimeIndex)):
            return df

        if isinstance(df.index, pd.PeriodIndex):
            new_df = self._coerce_period_to_datetime_index(df)
        elif pd.api.types.is_integer_dtype(df.index):
            try:
                new_df = self._coerce_int_to_range_index(df)
            except AssertionError:
                raise ValueError(
                    f"Coercion of index of {input_var} from integer pd.Index to "
                    "pd.RangeIndex failed. Please ensure that indexes are equally "
                    "spaced apart."
                )
        else:
            raise ValueError(
                f"{input_var} must have one of the following index types: "
                "pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, pd.Index"
                f"(int dtype). Found index of type: {type(df.index)}"
            )
        return new_df

    def _fit(
        self: "SkforecastAutoreg",
        y: pd.Series,
        X: Optional[pd.DataFrame],
        fh: Optional[ForecastingHorizon],
    ):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here.
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        del fh  # avoid being detected as unused by ``vulture`` like tools

        self._forecaster = self._create_forecaster()

        # skforecast does not support PeriodIndex and Integer Index.
        # So converting to supported index types here if necessary.
        y_new = self._make_index_compatible(y, "y")
        X_new = self._make_index_compatible(X, "X")

        self._forecaster.fit(y_new, exog=self._coerce_column_names(X_new))

        return self

    def _get_horizon_details(
        self: "SkforecastAutoreg", fh: Optional[ForecastingHorizon]
    ):
        if not fh.is_all_out_of_sample(self.cutoff):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support in-sample predictions."
            )

        out_of_sample_horizon = fh.to_out_of_sample(self.cutoff)
        maximum_forecast_horizon = out_of_sample_horizon.to_relative(self.cutoff)[-1]

        absolute_horizons = out_of_sample_horizon.to_absolute_index(self.cutoff)
        horizon_positions = out_of_sample_horizon.to_indexer(self.cutoff)

        return maximum_forecast_horizon, absolute_horizons, horizon_positions

    def _predict(
        self: "SkforecastAutoreg",
        fh: Optional[ForecastingHorizon],
        X: Optional[pd.DataFrame],
    ):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions
        """
        (
            maximum_forecast_horizon,
            absolute_horizons,
            horizon_positions,
        ) = self._get_horizon_details(fh)

        X_new = self._make_index_compatible(X, "X")

        point_predictions = self._forecaster.predict(
            maximum_forecast_horizon, exog=self._coerce_column_names(X_new)
        ).to_numpy()
        final_point_predictions = pd.Series(
            point_predictions[horizon_positions],
            index=absolute_horizons,
            name=None if self._y.name is None else str(self._y.name),
        )

        return final_point_predictions

    def _predict_quantiles(
        self: "SkforecastAutoreg",
        fh: Optional[ForecastingHorizon],
        X: Optional[pd.DataFrame],
        alpha: list[float],
    ):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        (
            maximum_forecast_horizon,
            absolute_horizons,
            horizon_positions,
        ) = self._get_horizon_details(fh)

        var_names = self._get_varnames()
        var_name = var_names[0]

        quantile_predictions_indices = pd.MultiIndex.from_product([var_names, alpha])
        quantile_predictions = pd.DataFrame(
            index=absolute_horizons, columns=quantile_predictions_indices
        )

        X_new = self._make_index_compatible(X, "X")

        quantile_pred = self._forecaster.predict_quantiles(
            maximum_forecast_horizon,
            exog=self._coerce_column_names(X_new),
            quantiles=alpha,
        )

        for quantile in alpha:
            quantile_predictions[(var_name, quantile)] = (
                quantile_pred[f"q_{quantile}"].iloc[horizon_positions].to_list()
            )
        return quantile_predictions

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        param1 = {
            "regressor": LinearRegression(),
            "lags": 2,
            "transformer_exog": StandardScaler(),
        }
        param2 = {
            "regressor": RandomForestRegressor(),
            "lags": [1, 3],
            "differentiation": 2,
        }

        return [param1, param2]


__all__ = ["SkforecastAutoreg"]
