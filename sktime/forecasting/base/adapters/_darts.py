# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for Darts models."""

import abc
from typing import Optional, Union

import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.warnings import warn

__author__ = ["yarnabrina", "fnhirwa"]

LAGS_TYPE = Optional[Union[int, list[int], dict[str, Union[int, list[int]]]]]
PAST_LAGS_TYPE = Optional[Union[int, list[int], dict[str, Union[int, list[int]]]]]
FUTURE_LAGS_TYPE = Optional[
    Union[tuple[int, int], list[int], dict[str, Union[tuple[int, int], list[int]]]]
]


class _DartsRegressionAdapter(BaseForecaster):
    """Base adapter class for Darts Regression Model.

    Parameters
    ----------
    lags
        Lagged target values used to predict the next time step. If an integer is given
        the last `lags` past lags are used (from -1 backward). Otherwise a list of
        integers with lags is required (each lag must be < 0). If a dictionary is given,
        keys correspond to the component names
        (of first series when using multiple series) and
        the values correspond to the component lags(integer or list of integers).
    lags_past_covariates
        Number of lagged past_covariates values used to predict the next time step. If
        an integer is given the last `lags_past_covariates` past lags are used
        (inclusive, starting from lag -1). Otherwise a list of integers
        with lags < 0 is required. If a dictionary is given, keys correspond to the
        past_covariates component names(of first series when using multiple series)
        and the values correspond to the component lags(integer or list of integers).
    lags_future_covariates
        Number of lagged future_covariates values used to predict the next time step. If
        a tuple (past, future) is given the last `past` lags in the past are used
        (inclusive, starting from lag -1) along with the first `future` future lags
        (starting from 0 - the prediction time - up to `future - 1` included). Otherwise
        a list of integers with lags is required. If dictionary is given,
        keys correspond to the future_covariates component names
        (of first series when using multiple series) and the values
        correspond to the component lags(integer or list of integers).
    output_chunk_shift
        Optional, the number of steps to shift the start of the output chunk into the
        future (relative to the input chunk end). This will create a gap between the
        input (history of target and past covariates) and output. If the model supports
        future_covariates, the lags_future_covariates are relative to the first step in
        the shifted output chunk. Predictions will start output_chunk_shift steps after
        the end of the target series. If output_chunk_shift is set, the model cannot
        generate autoregressive predictions (n > output_chunk_length).
    output_chunk_length
        Number of time steps predicted at once by the internal regression model. Does
        not have to equal the forecast horizon `n` used in `predict()`. However, setting
        `output_chunk_length` equal to the forecast horizon may be useful if the
        covariates don't extend far enough into the future.
    add_encoders
        A large number of past and future covariates can be automatically generated with
        `add_encoders`. This can be done by adding multiple pre-defined index encoders
        and/or custom user-made functions that will be used as index encoders.
        Additionally, a transformer such as Darts' :class:`Scaler` can be added to
        transform the generated covariates. This happens all under one hood and only
        needs to be specified at model creation. Read
        :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>`
        to find out more about ``add_encoders``. Default: ``None``. An example showing
        some of ``add_encoders`` features:

        .. highlight:: python
        .. code-block:: python

            add_encoders={
                'cyclic': {'future': ['month']},
                'datetime_attribute': {'future': ['hour', 'dayofweek']},
                'position': {'past': ['relative'], 'future': ['relative']},
                'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                'transformer': Scaler()
            }
        ..
    model
        Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible
        to use model that doesn't support multi-output regression for multivariate
        timeseries, in which case one regressor will be used per component in the
        multivariate series. If None, defaults to:
        ``sklearn.linear_model.LinearRegression(n_jobs=-1)``.
    multi_models
        If True, a separate model will be trained for each future lag to predict. If
        False, a single model is trained to predict at step 'output_chunk_length' in the
        future. Default: True.
    use_static_covariates
        Whether the model should use static covariate information in case the input
        `series` passed to ``fit()`` contain static covariates. If ``True``, and static
        covariates are available at fitting time, will enforce that all target `series`
        have the same static covariate dimensionality in ``fit()`` and ``predict()``.

    past_covariates : Optional[List[str]], optional
        column names in ``X`` which are known only for historical data, by default None
    num_samples : Optional[int], optional
        Number of times a prediction is sampled from a probabilistic model, by default
        1000

    Notes
    -----
    If unspecified, all columns will be assumed to be known during prediction duration.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["yarnabrina", "fnhirwa"],
        "maintainers": ["yarnabrina", "fnhirwa"],
        "python_version": ">=3.9",
        "python_dependencies": [["u8darts>=0.29", "darts>=0.29"]],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": False,
    }

    def __init__(
        self: "_DartsRegressionAdapter",
        lags: LAGS_TYPE = None,
        lags_past_covariates: PAST_LAGS_TYPE = None,
        lags_future_covariates: FUTURE_LAGS_TYPE = None,
        output_chunk_length: Optional[int] = 1,
        output_chunk_shift: Optional[int] = 0,
        add_encoders: Optional[dict] = None,
        model=None,
        multi_models: Optional[bool] = True,
        use_static_covariates: Optional[bool] = True,
        past_covariates: Optional[list[str]] = None,
        num_samples: Optional[int] = 1000,
    ) -> None:
        self.lags = lags
        self.lags_past_covariates = lags_past_covariates
        self.lags_future_covariates = lags_future_covariates
        self.output_chunk_length = output_chunk_length
        self.output_chunk_shift = output_chunk_shift
        self.add_encoders = add_encoders
        self.model = model
        self.multi_models = multi_models
        self.use_static_covariates = use_static_covariates

        if past_covariates is not None and not isinstance(past_covariates, list):
            raise TypeError(
                f"Expected past_covariates to be a list, found {type(past_covariates)}."
            )
        self.past_covariates = past_covariates
        if num_samples is not None and not isinstance(num_samples, int):
            raise TypeError(
                f"Expected num_samples to be an integer, found {type(num_samples)}."
            )
        self.num_samples = num_samples

        super().__init__()

        # initialize internal variables to avoid AttributeError
        self._forecaster = None

    @staticmethod
    def convert_dataframe_to_timeseries(dataset: pd.DataFrame):
        """Convert dataset for compatibility with ``darts``.

        Parameters
        ----------
        dataset : pandas.DataFrame
            source dataset to convert from

        Returns
        -------
        darts.TimeSeries
            converted target dataset
        """
        import darts

        dataset_copy = _handle_input_index(dataset)
        return darts.TimeSeries.from_dataframe(dataset_copy)

    def convert_exogenous_dataset(
        self: "_DartsRegressionModelsAdapter", dataset: Optional[pd.DataFrame]
    ):
        """Make exogenous features to ``darts`` compatible, if available.

        Parameters
        ----------
        dataset : Optional[pandas.DataFrame]
            available data on exogenous features

        Returns
        -------
        Tuple[darts.TimeSeries, darts.TimeSeries]
            converted data on future known and future unknown exogenous features
        """
        if dataset is None and self.past_covariates:
            raise ValueError(
                f"Expected following exogenous features: {self.past_covariates}."
            )

        if dataset is None:
            future_known_dataset = None
            future_unknown_dataset = None

        elif self.past_covariates is not None and self.lags_past_covariates is not None:
            future_unknown_dataset = self.convert_dataframe_to_timeseries(
                dataset[self.past_covariates]
            )
            future_known_dataset = self.convert_dataframe_to_timeseries(
                dataset.drop(columns=self.past_covariates)
            )
        elif self.lags_future_covariates is not None:
            future_unknown_dataset = None
            future_known_dataset = self.convert_dataframe_to_timeseries(dataset)
        else:
            future_known_dataset = None
            future_unknown_dataset = None

        return future_known_dataset, future_unknown_dataset

    @classmethod
    @abc.abstractmethod
    def _create_forecaster(self: "_DartsRegressionModelsAdapter"):
        """Create Darts model."""

    def _fit(
        self: "_DartsRegressionModelsAdapter",
        y: pd.DataFrame,
        X: Optional[pd.DataFrame],
        fh: Optional[ForecastingHorizon],
    ):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.DataFrame
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            For darts models `fh` is not used,
            the steps ahead for prediction is determined by `output_chunk_length`.
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        del fh  # avoid being detected as unused by ``vulture`` like tools
        endogenous_actuals = self.convert_dataframe_to_timeseries(y)
        unknown_exogenous, known_exogenous = self.convert_exogenous_dataset(X)
        # single-target variable for univariate prediction
        if endogenous_actuals.width > 1 and self.get_tag("scitype:y") == "univariate":
            raise ValueError(
                "Multi-target prediction is not supported by the quantile loss."
                " Please provide a single-target variable."
            )
        self._forecaster = self._create_forecaster()
        self._forecaster.fit(
            endogenous_actuals,
            past_covariates=unknown_exogenous,
            future_covariates=known_exogenous,
        )

        return self

    def _predict(
        self: "_DartsRegressionModelsAdapter",
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
            The forecasting horizon value should be less than the value
            of ``output_chunk_length`` fitted to the model, otherwise the prediction
            result will be from auto-regression.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions
        """
        if not fh.is_all_out_of_sample(cutoff=self.cutoff):
            raise NotImplementedError("in-sample prediction is currently not supported")
        non_auto_regressive_fh = ForecastingHorizon(
            [i for i in range(1, self.output_chunk_length + 1)]
        )
        # warning for fh out of range of output_chunk_length
        if max(fh.to_indexer(cutoff=self.cutoff)) > max(
            non_auto_regressive_fh.to_indexer(cutoff=self.cutoff)
        ):
            warn(
                f"Forecasting horizon values: {fh} are out of range of"
                " output_chunk_length. The prediction will be auto-regression based.",
                obj=self,
                stacklevel=2,
            )
        self.check_is_fitted()
        unknown_exogenous, known_exogenous = self.convert_exogenous_dataset(X)
        absolute_fh = fh.to_absolute(self.cutoff)
        maximum_forecast_horizon = fh.to_relative(self.cutoff)[-1]

        endogenous_point_predictions = self._forecaster.predict(
            maximum_forecast_horizon,
            past_covariates=unknown_exogenous,
            future_covariates=known_exogenous,
            num_samples=1,
        )
        expected_index = fh.get_expected_pred_idx(self.cutoff)
        abs_idx = absolute_fh.to_pandas().astype(expected_index.dtype)

        endogenous_point_predictions = endogenous_point_predictions.pd_dataframe()

        if _is_int64_type(expected_index):
            if X is not None:
                from pandas.core.indexes.numeric import Int64Index

                endogenous_point_predictions.index = Int64Index(
                    endogenous_point_predictions.index
                )

        if isinstance(expected_index, pd.PeriodIndex):
            endogenous_point_predictions.index = (
                endogenous_point_predictions.index.to_period(expected_index.freqstr)
            )
        if isinstance(expected_index, pd.RangeIndex):
            endogenous_point_predictions.index = pd.RangeIndex(
                start=0, stop=len(endogenous_point_predictions)
            )
        if isinstance(expected_index, pd.DatetimeIndex):
            endogenous_point_predictions.index = pd.date_range(
                start=expected_index[0],
                periods=len(endogenous_point_predictions),
                freq=expected_index.freq,
            )

        if (
            len(endogenous_point_predictions.columns) > 1
            and self._y.columns.dtype != "object"
        ):
            endogenous_point_predictions.columns = pd.RangeIndex(
                start=0, stop=len(endogenous_point_predictions.columns), step=1
            )
        else:
            endogenous_point_predictions.columns = [
                "c" + str(i) for i in range(endogenous_point_predictions.shape[1])
            ]
        return endogenous_point_predictions.loc[abs_idx]


class _DartsRegressionModelsAdapter(_DartsRegressionAdapter):
    """Adapter class for Darts Regression models.

    Parameters
    ----------
    lags
        Lagged target values used to predict the next time step. If an integer is given
        the last `lags` past lags are used (from -1 backward). Otherwise a list of
        integers with lags is required (each lag must be < 0). If a dictionary is given,
        keys correspond to the component names
        (of first series when using multiple series) and
        the values correspond to the component lags(integer or list of integers).
    lags_past_covariates
        Number of lagged past_covariates values used to predict the next time step. If
        an integer is given the last `lags_past_covariates` past lags are used
        (inclusive, starting from lag -1). Otherwise a list of integers
        with lags < 0 is required. If a dictionary is given, keys correspond to the
        past_covariates component names(of first series when using multiple series)
        and the values correspond to the component lags(integer or list of integers).
    lags_future_covariates
        Number of lagged future_covariates values used to predict the next time step. If
        a tuple (past, future) is given the last `past` lags in the past are used
        (inclusive, starting from lag -1) along with the first `future` future lags
        (starting from 0 - the prediction time - up to `future - 1` included). Otherwise
        a list of integers with lags is required. If dictionary is given,
        keys correspond to the future_covariates component names
        (of first series when using multiple series) and the values
        correspond to the component lags(integer or list of integers).
    output_chunk_length
        Number of time steps predicted at once by the internal regression model. Does
        not have to equal the forecast horizon `n` used in `predict()`. However, setting
        `output_chunk_length` equal to the forecast horizon may be useful if the
        covariates don't extend far enough into the future.
    add_encoders
        A large number of past and future covariates can be automatically generated with
        `add_encoders`. This can be done by adding multiple pre-defined index encoders
        and/or custom user-made functions that will be used as index encoders.
        Additionally, a transformer such as Darts' :class:`Scaler` can be added to
        transform the generated covariates. This happens all under one hood and only
        needs to be specified at model creation. Read
        :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>`
        to find out more about ``add_encoders``. Default: ``None``. An example showing
        some of ``add_encoders`` features:

        .. highlight:: python
        .. code-block:: python

            add_encoders={
                'cyclic': {'future': ['month']},
                'datetime_attribute': {'future': ['hour', 'dayofweek']},
                'position': {'past': ['relative'], 'future': ['relative']},
                'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                'transformer': Scaler()
            }
        ..
    model
        Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible
        to use model that doesn't support multi-output regression for multivariate
        timeseries, in which case one regressor will be used per component in the
        multivariate series. If None, defaults to:
        ``sklearn.linear_model.LinearRegression(n_jobs=-1)``.
    multi_models
        If True, a separate model will be trained for each future lag to predict. If
        False, a single model is trained to predict at step 'output_chunk_length' in the
        future. Default: True.
    use_static_covariates
        Whether the model should use static covariate information in case the input
        `series` passed to ``fit()`` contain static covariates. If ``True``, and static
        covariates are available at fitting time, will enforce that all target `series`
        have the same static covariate dimensionality in ``fit()`` and ``predict()``.

    past_covariates : Optional[List[str]], optional
        column names in ``X`` which are known only for historical data, by default None
    num_samples : Optional[int], optional
        Number of times a prediction is sampled from a probabilistic model, by default
        1000

    Notes
    -----
    If unspecified, all columns will be assumed to be known during prediction duration.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["yarnabrina", "fnhirwa"],
        "maintainers": ["yarnabrina", "fnhirwa"],
        "python_version": ">=3.9",
        "python_dependencies": [["u8darts>=0.29", "darts>=0.29"]],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:insample": False,
    }

    def __init__(
        self: "_DartsRegressionModelsAdapter",
        lags: LAGS_TYPE = None,
        lags_past_covariates: PAST_LAGS_TYPE = None,
        lags_future_covariates: FUTURE_LAGS_TYPE = None,
        output_chunk_length: Optional[int] = 1,
        output_chunk_shift: Optional[int] = 0,
        add_encoders: Optional[dict] = None,
        multi_models: Optional[bool] = True,
        use_static_covariates: Optional[bool] = True,
        past_covariates: Optional[list[str]] = None,
        num_samples: Optional[int] = 1000,
    ) -> None:
        self.lags = lags
        self.lags_past_covariates = lags_past_covariates
        self.lags_future_covariates = lags_future_covariates
        self.output_chunk_length = output_chunk_length
        self.output_chunk_shift = output_chunk_shift
        self.add_encoders = add_encoders
        self.multi_models = multi_models
        self.use_static_covariates = use_static_covariates
        self.past_covariates = past_covariates
        self.num_samples = num_samples
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=add_encoders,
            model=None,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            past_covariates=past_covariates,
            num_samples=num_samples,
        )

        # initialize internal variables to avoid AttributeError
        self._forecaster = None

    def _predict_quantiles(
        self,
        fh: Optional[ForecastingHorizon],
        X: Optional[pd.DataFrame],
        alpha: list[float],
    ):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and default _predict_interval

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        alpha : list of float, optional (default=[0.5])
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
        non_auto_regressive_fh = ForecastingHorizon(
            [i for i in range(1, self.output_chunk_length + 1)]
        )
        # warning for fh out of range of output_chunk_length
        if max(fh.to_indexer(cutoff=self.cutoff)) > max(
            non_auto_regressive_fh.to_indexer(cutoff=self.cutoff)
        ):
            warn(
                f"Forecasting horizon values: {fh} are out of range of"
                " output_chunk_length. The prediction will be auto-regression based.",
                obj=self,
                stacklevel=2,
            )
        unknown_exogenous, known_exogenous = self.convert_exogenous_dataset(X)
        maximum_forecast_horizon = fh.to_relative(self.cutoff)[-1]
        absolute_fh = fh.to_absolute(self.cutoff)
        endogenous_quantile_predictions = self._forecaster.predict(
            maximum_forecast_horizon,
            past_covariates=unknown_exogenous,
            future_covariates=known_exogenous,
            num_samples=self.num_samples,
        ).quantiles_df(quantiles=alpha)
        variable_names = self._get_varnames()
        multi_index = pd.MultiIndex.from_product(
            [variable_names, alpha], names=["variable", "quantile"]
        )
        expected_index = fh.get_expected_pred_idx(self.cutoff)
        endogenous_quantile_predictions.index = (
            endogenous_quantile_predictions.index.astype(expected_index.dtype)
        )

        abs_idx = absolute_fh.to_pandas().astype(expected_index.dtype)
        endogenous_quantile_predictions.columns = multi_index
        return endogenous_quantile_predictions.loc[abs_idx]


def _handle_input_index(dataset: pd.DataFrame) -> pd.DataFrame:
    """Convert input dataset index to the compatible type for ``darts``.

    Parameters
    ----------
    dataset: pandas.DataFrame
        dataset with index to be converted

    Returns
    -------
    pandas.DataFrame
        converted dataset
    """
    if isinstance(dataset.index, pd.RangeIndex):
        return dataset
    dataset_copy = dataset.copy(deep=True)

    if isinstance(dataset_copy.index, (pd.DatetimeIndex, pd.RangeIndex)):
        dataset_copy.index = pd.date_range(
            start=dataset_copy.index[0],
            periods=len(dataset_copy),
            freq=dataset_copy.index.freq,
        )
        return dataset_copy

    if isinstance(dataset_copy.index, pd.PeriodIndex):
        dataset_copy.index = dataset_copy.index.to_timestamp()
        return dataset_copy

    if _is_int64_type(dataset_copy.index):
        dataset_copy.index = pd.RangeIndex(
            start=dataset_copy.index.min(),
            stop=dataset_copy.index.max() + 1,
            step=dataset_copy.index[1] - dataset_copy.index[0],
        )
        return dataset_copy

    return dataset_copy


def _is_int64_type(index: pd.Index) -> bool:
    """Check if the index is an Int64Index type for pandas older versions.

    Parameters
    ----------
    index : pd.Index
        Index to check

    Returns
    -------
    bool
        True if the index is numeric, False otherwise
    """
    try:
        from pandas.core.indexes.numeric import Int64Index

        return isinstance(index, Int64Index)
    except ImportError:
        return False


__all__ = ["_DartsRegressionAdapter", "_DartsRegressionModelsAdapter"]
