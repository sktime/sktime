# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for NeuralForecast models."""
import abc
import functools
import typing

import pandas

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

__all__ = ["_NeuralForecastAdapter"]
__author__ = ["yarnabrina"]


class _NeuralForecastAdapter(BaseForecaster):
    """Base adapter class for NeuralForecast models.

    Parameters
    ----------
    freq : str (default="auto")
        frequency of the data, see available frequencies [1]_ from ``pandas``

        default ("auto") interprets freq from ForecastingHorizon in ``fit``
    local_scaler_type : str (default=None)
        scaler to apply per-series to all features before fitting, which is inverted
        after predicting

        can be one of the following:

        - 'standard'
        - 'robust'
        - 'robust-iqr'
        - 'minmax'
        - 'boxcox'
    futr_exog_list : str list, (default=None)
        future exogenous variables
    verbose_fit : bool (default=False)
        print processing steps during fit
    verbose_predict : bool (default=False)
        print processing steps during predict

    Notes
    -----
    Only ``futr_exog_list`` will be considered as exogenous variables.

    References
    ----------
    .. [1] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": ["yarnabrina"],
        "maintainers": ["yarnabrina"],
        "python_version": ">=3.8",
        "python_dependencies": ["neuralforecast"],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "handles-missing-data": False,
        "capability:insample": False,
    }

    def __init__(
        self: "_NeuralForecastAdapter",
        freq: str = "auto",
        local_scaler_type: typing.Optional[
            typing.Literal["standard", "robust", "robust-iqr", "minmax", "boxcox"]
        ] = None,
        futr_exog_list: typing.Optional[typing.List[str]] = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
    ) -> None:
        self.freq = freq
        self.local_scaler_type = local_scaler_type

        self.futr_exog_list = futr_exog_list

        self.verbose_fit = verbose_fit
        self.verbose_predict = verbose_predict

        super().__init__()

        # initiate internal variables to avoid AttributeError in future
        self._freq = None

        self.id_col = "unique_id"
        self.time_col = "ds"
        self.target_col = "y"

        self.needs_X = self.algorithm_exogenous_support and bool(self.futr_exog_list)

        self.set_tags(**{"ignores-exogeneous-X": not self.needs_X})

    @functools.cached_property
    @abc.abstractmethod
    def algorithm_exogenous_support(self: "_NeuralForecastAdapter") -> bool:
        """Set support for exogenous features."""

    @functools.cached_property
    @abc.abstractmethod
    def algorithm_name(self: "_NeuralForecastAdapter") -> str:
        """Set custom model name."""

    @functools.cached_property
    @abc.abstractmethod
    def algorithm_class(self: "_NeuralForecastAdapter"):
        """Import underlying NeuralForecast algorithm class."""

    @functools.cached_property
    @abc.abstractmethod
    def algorithm_parameters(self: "_NeuralForecastAdapter") -> dict:
        """Get keyword parameters for the underlying NeuralForecast algorithm class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class

        Notes
        -----
        This method should not include following parameters:

        - future exogenous columns (``futr_exog_list``) - used from ``__init__``
        - historical exogenous columns (``hist_exog_list``) - not supported
        - statis exogenous columns (``stat_exog_list``) - not supported
        - custom model name (``alias``) - used from ``algorithm_name``
        """

    def _instantiate_model(self: "_NeuralForecastAdapter", fh: ForecastingHorizon):
        """Instantiate the model."""
        exogenous_parameters = (
            {"futr_exog_list": self.futr_exog_list} if self.needs_X else {}
        )

        algorithm_instance = self.algorithm_class(
            fh,
            alias=self.algorithm_name,
            **self.algorithm_parameters,
            **exogenous_parameters,
        )

        from neuralforecast import NeuralForecast

        model = NeuralForecast(
            [algorithm_instance], self._freq, local_scaler_type=self.local_scaler_type
        )

        return model

    def _fit(
        self: "_NeuralForecastAdapter",
        y: pandas.Series,
        X: typing.Optional[pandas.DataFrame],
        fh: ForecastingHorizon,
    ) -> "_NeuralForecastAdapter":
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to have a single column/variable
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to have at least one column/variable
            Exogeneous time series to fit to.

        Returns
        -------
        self : _NeuralForecastAdapter
            reference to self

        Raises
        ------
        ValueError
            When ``freq="auto"`` and cannot be interpreted from ``ForecastingHorizon``
        """
        if not fh.is_all_out_of_sample(cutoff=self.cutoff):
            raise NotImplementedError("in-sample prediction is currently not supported")

        if self.freq == "auto" and fh.freq is None:
            # when freq cannot be interpreted from ForecastingHorizon
            raise ValueError(
                f"Error in {self.__class__.__name__}, "
                f"could not interpret freq, "
                f"try passing freq in model initialization"
            )

        self._freq = fh.freq if self.freq == "auto" else self.freq

        train_indices = y.index
        if isinstance(train_indices, pandas.PeriodIndex):
            train_indices = train_indices.to_timestamp(freq=self._freq)

        train_data = {
            self.id_col: 1,
            self.time_col: train_indices.to_numpy(),
            self.target_col: y.to_numpy(),
        }

        if self.futr_exog_list and X is None:
            raise ValueError("Missing exogeneous data, 'futr_exog_list' is non-empty.")

        if self.futr_exog_list:
            for column in self.futr_exog_list:
                train_data[column] = X[column].to_numpy()

        train_dataset = pandas.DataFrame(data=train_data)

        maximum_forecast_horizon = fh.to_relative(self.cutoff)[-1]
        self._forecaster = self._instantiate_model(maximum_forecast_horizon)

        self._forecaster.fit(df=train_dataset, verbose=self.verbose_fit)

        return self

    def _predict(
        self: "_NeuralForecastAdapter",
        fh: typing.Optional[ForecastingHorizon],
        X: typing.Optional[pandas.DataFrame],
    ) -> pandas.Series:
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
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            guaranteed to have a single column/variable
            Point predictions

        Notes
        -----
        This method does not use ``fh``, the one passed during ``fit`` takes precedence.
        """
        del fh  # to avoid being detected as unused by ``vulture`` etc.

        predict_parameters: dict = {"verbose": self.verbose_predict}

        # this block is probably unnecessary, but kept to be safe
        # the check in fit ensures X is passed if futr_exog_list is non-empty
        # base framework should ensure X is passed in predict in that case
        if self.futr_exog_list and X is None:
            raise ValueError("Missing exogeneous data, 'futr_exog_list' is non-empty.")

        if self.futr_exog_list:
            predict_indices = X.index
            if isinstance(predict_indices, pandas.PeriodIndex):
                predict_indices = predict_indices.to_timestamp(freq=self._freq)

            predict_data = {self.id_col: 1, self.time_col: predict_indices.to_numpy()}

            for column in self.futr_exog_list:
                predict_data[column] = X[column].to_numpy()

            predict_dataset = pandas.DataFrame(data=predict_data)

            predict_parameters["futr_df"] = predict_dataset

        model_forecasts = self._forecaster.predict(**predict_parameters)

        prediction_column_names = [
            column
            for column in model_forecasts.columns
            if column.startswith(self.algorithm_name)
        ]

        # this block is necessary only for specific values of ``loss``
        # for example, when using ``MQLoss`` for multiple quantiles
        if len(prediction_column_names) > 1:
            raise NotImplementedError("Multiple prediction columns are not supported.")

        model_point_predictions = model_forecasts[prediction_column_names[0]].to_numpy()

        absolute_horizons = self.fh.to_absolute_index(self.cutoff)
        horizon_positions = self.fh.to_indexer(self.cutoff)

        final_predictions = pandas.Series(
            model_point_predictions[horizon_positions],
            index=absolute_horizons,
            name=self._y.name,
        )

        return final_predictions
