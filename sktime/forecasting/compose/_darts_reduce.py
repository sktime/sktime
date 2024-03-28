# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for Darts regression models."""
from typing import Dict, List, Optional, Tuple, Type, Union

import pandas

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

__all__ = ["DartsRegressionModel"]
__author__ = ["yarnabrina"]

LAGS_TYPE = Union[int, List[int], Dict[str, Union[int, List[int]]]]
FUTURE_LAGS_TYPE = Union[
    Tuple[int, int],
    List[int],
    Dict[str, Union[Tuple[int, int], List[int]]],
]


class DartsRegressionModel(BaseForecaster):
    """Base adapter class for Darts [1]_ regression models [2]_ .

    Parameters
    ----------
    past_covariates
        Optionally, column names containing past-observed covariates
    future_covariates
        Optionally, column names containing future-known covariates
    static_covariates
        Optionally, column names containing constant covariates
    lags
        Lagged target `series` values used to predict the next time step/s. If an
        integer, must be > 0. Uses the last `n=lags` past lags; e.g.
        `(-1, -2, ..., -lags)`, where `0` corresponds the first predicted time step of
        each sample. If a list of integers, each value must be < 0. Uses only the
        specified values as lags. If a dictionary, the keys correspond to the `series`
        component names (of the first series when using multiple series) and the values
        correspond to the component lags (integer or list of integers). The key
        'default_lags' can be used to provide default lags for un-specified components.
        Raises and error if some components are missing and the 'default_lags' key is
        not provided.
    lags_past_covariates
        Lagged `past_covariates` values used to predict the next time step/s. If an
        integer, must be > 0. Uses the last `n=lags_past_covariates` past lags; e.g.
        `(-1, -2, ..., -lags)`, where `0` corresponds to the first predicted time step
        of each sample. If a list of integers, each value must be < 0. Uses only the
        specified values as lags. If a dictionary, the keys correspond to the
        `past_covariates` component names (of the first series when using multiple
        series) and the values correspond to the component lags (integer or list of
        integers). The key 'default_lags' can be used to provide default lags for
        un-specified components. Raises and error if some components are missing and the
        'default_lags' key is not provided.
    lags_future_covariates
        Lagged `future_covariates` values used to predict the next time step/s. If a
        tuple of `(past, future)`, both values must be > 0. Uses the last `n=past` past
        lags and `n=future` future lags; e.g.
        `(-past, -(past - 1), ..., -1, 0, 1, .... future - 1)`, where `0` corresponds
        the first predicted time step of each sample. If a list of integers, uses only
        the specified values as lags. If a dictionary, the keys correspond to the
        `future_covariates` component names (of the first series when using multiple
        series) and the values correspond to the component lags (tuple or list of
        integers). The key 'default_lags' can be used to provide default lags for
        un-specified components. Raises and error if some components are missing and the
        'default_lags' key is not provided.
    output_chunk_length
        Number of time steps predicted at once (per chunk) by the internal model. It is
        not the same as forecast horizon `n` used in `predict()`, which is the desired
        number of prediction points generated using a one-shot- or auto-regressive
        forecast. Setting `n <= output_chunk_length` prevents auto-regression. This is
        useful when the covariates don't extend far enough into the future, or to
        prohibit the model from using future values of past and / or future covariates
        for prediction (depending on the model's covariate support).
    add_encoders
        A large number of past and future covariates can be automatically generated with
        `add_encoders`. This can be done by adding multiple pre-defined index encoders
        and/or custom user-made functions that will be used as index encoders.
        Additionally, a transformer such as Darts' :class:`Scaler` can be added to
        transform the generated covariates. This happens all under one hood and only
        needs to be specified at model creation. Read
        :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to
        find out more about ``add_encoders``. Default: ``None``. An example showing some
        of ``add_encoders`` features:

        .. highlight:: python
        .. code-block:: python

            def encode_year(idx):
                return (idx.year - 1950) / 50

            add_encoders={
                'cyclic': {'future': ['month']},
                'datetime_attribute': {'future': ['hour', 'dayofweek']},
                'position': {'past': ['relative'], 'future': ['relative']},
                'custom': {'past': [encode_year]},
                'transformer': Scaler(),
                'tz': 'CET'
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
    max_samples_per_ts
        This is an integer upper bound on the number of tuples that can be produced per
        time series. It can be used in order to have an upper bound on the total size of
        the dataset and ensure proper sampling. If `None`, it will read all of the
        individual time series in advance (at dataset creation) to know their sizes,
        which might be expensive on big datasets. If some series turn out to have a
        length that would allow more than `max_samples_per_ts`, only the most recent
        `max_samples_per_ts` samples will be considered.
    n_jobs_multioutput_wrapper
        Number of jobs of the MultiOutputRegressor wrapper to run in parallel. Only used
        if the model doesn't support multi-output regression natively.
    fit_kwargs
        Additional keyword arguments passed to the `fit` method of the model.
    num_samples : int, default: 1
        Number of times a prediction is sampled from a probabilistic model. Should be
        set to 1 for deterministic models.
    verbose
        Optionally, whether to print progress.
    predict_likelihood_parameters
        If set to `True`, the model predict the parameters of its Likelihood parameters
        instead of the target. Only supported for probabilistic models with a
        likelihood, `num_samples = 1` and `n<=output_chunk_length`. Default: ``False``
    show_warnings
        Optionally, control whether warnings are shown. Not effective for all models.
    predict_kwargs
        Additional keyword arguments passed to the `predict` method of the model. Only
        works with univariate target series.

    Examples
    --------
    Import necessary libraries and modules

    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> from sktime.datasets import load_airline, load_longley
    >>> from sktime.forecasting.compose import DartsRegressionModel

    Load datasets

    >>> airline_y = load_airline()
    >>> longley_y, longley_X = load_longley()

    Create sklearn-compatible regressors

    >>> lm_regressor = LinearRegression()
    >>> rf_regressor = RandomForestRegressor(random_state=0)

    Forecast univariate endogenous variable

    >>> forecaster_1 = DartsRegressionModel(  # doctest: +SKIP
    ...     lags=12, model=lm_regressor
    ... )
    >>> forecaster_1.fit(airline_y)  # doctest: +SKIP
    DartsRegressionModel(lags=12, model=LinearRegression())
    >>> forecaster_1.predict(fh=[2, 4, 5])  # doctest: +SKIP
    1961-02    429.138107
    1961-04    490.962074
    1961-05    527.765278
    Freq: M, Name: Number of airline passengers, dtype: float64

    Forecast multivariate endogenous variable

    >>> forecaster_2 = DartsRegressionModel(  # doctest: +SKIP
    ...     lags=[-1, -3], model=rf_regressor
    ... )
    >>> forecaster_2.fit(longley_X)  # doctest: +SKIP
    DartsRegressionModel(lags=[-1, -3], model=RandomForestRegressor(random_state=0))
    >>> forecaster_2.predict(fh=range(1, 3))  # doctest: +SKIP
          GNPDEFL        GNP    UNEMP    ARMED        POP
    1963  115.742  530416.58  4246.24  2714.15  128288.76
    1964  116.008  533610.54  4286.63  2704.55  128566.83

    References
    ----------
    .. [1] https://unit8co.github.io/darts/
    .. [2] https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_model.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": ["yarnabrina"],
        "maintainers": ["yarnabrina"],
        "python_version": ">=3.8",
        "python_dependencies": ["u8darts"],
        "python_dependencies_alias": {"u8darts": "darts"},
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "both",
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": [pandas.DatetimeIndex, pandas.RangeIndex],
        "handles-missing-data": False,
        "capability:insample": False,
    }

    def __init__(
        self: "DartsRegressionModel",
        past_covariates: Optional[List[str]] = None,
        future_covariates: Optional[List[str]] = None,
        static_covariates: Optional[List[str]] = None,
        lags: Optional[LAGS_TYPE] = None,
        lags_past_covariates: Optional[LAGS_TYPE] = None,
        lags_future_covariates: Optional[FUTURE_LAGS_TYPE] = None,
        output_chunk_length: int = 1,
        add_encoders: Optional[Dict] = None,
        model=None,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
        max_samples_per_ts: Optional[int] = None,
        n_jobs_multioutput_wrapper: Optional[int] = None,
        fit_kwargs: Optional[Dict] = None,
        num_samples: int = 1,
        verbose: bool = False,
        predict_likelihood_parameters: bool = False,
        show_warnings: bool = True,
        predict_kwargs: Optional[Dict] = None,
    ) -> None:
        # common parameters for fit+predict in sktime compatible form
        self.past_covariates = past_covariates
        self.future_covariates = future_covariates
        self.static_covariates = static_covariates

        # common parameters for __init__
        self.lags = lags
        self.lags_past_covariates = lags_past_covariates
        self.lags_future_covariates = lags_future_covariates
        self.output_chunk_length = output_chunk_length
        self.add_encoders = add_encoders
        self.model = model
        self.multi_models = multi_models
        self.use_static_covariates = use_static_covariates

        # common parameters for fit
        self.max_samples_per_ts = max_samples_per_ts
        self.n_jobs_multioutput_wrapper = n_jobs_multioutput_wrapper
        self.fit_kwargs = fit_kwargs

        # common parameters for predict
        self.num_samples = num_samples
        self.verbose = verbose
        self.predict_likelihood_parameters = predict_likelihood_parameters
        self.show_warnings = show_warnings
        self.predict_kwargs = predict_kwargs

        super().__init__()

        self._model = model
        self._forecaster = None

        self._past_covariates = (
            [] if self.past_covariates is None else self.past_covariates
        )
        self._future_covariates = (
            [] if self.future_covariates is None else self.future_covariates
        )
        self._static_covariates = (
            [] if self.static_covariates is None else self.static_covariates
        )

        self._fit_kwargs = {} if self.fit_kwargs is None else self.fit_kwargs
        self._predict_kwargs = (
            {} if self.predict_kwargs is None else self.predict_kwargs
        )

        self._non_constant_covariates = self._past_covariates + self._future_covariates

        self._all_covariates = self._non_constant_covariates + self._static_covariates

        self.needs_X = bool(self._all_covariates)

        self.set_tags(**{"ignores-exogeneous-X": not self.needs_X})

    def create_forecaster(self: "DartsRegressionModel"):
        """Create Darts model."""
        from darts.models import RegressionModel
        from sklearn.base import clone

        self._model = clone(self.model)

        model = RegressionModel(
            lags=self.lags,
            lags_past_covariates=self.lags_past_covariates,
            lags_future_covariates=self.lags_future_covariates,
            output_chunk_length=self.output_chunk_length,
            add_encoders=self.add_encoders,
            model=self.model,
            multi_models=self.multi_models,
            use_static_covariates=self.use_static_covariates,
        )

        return model

    @staticmethod
    def convert_dataset_index(dataset: pandas.DataFrame) -> pandas.DataFrame:
        """Convert dataset index for compatibility with ``darts``.

        Parameters
        ----------
        dataset : pandas.DataFrame
            dataset to convert from

        Returns
        -------
        pandas.DataFrame
            converted dataset
        """
        if isinstance(dataset.index, (pandas.DatetimeIndex, pandas.RangeIndex)):
            return dataset

        dataset_copy = dataset.copy(deep=True)

        if isinstance(dataset_copy.index, pandas.PeriodIndex):
            dataset_copy.index = dataset_copy.index.to_timestamp()

            return dataset_copy

        if pandas.api.types.is_integer_dtype(dataset_copy.index):
            # TODO: extend if there are gaps in the index
            dataset_copy.index = pandas.RangeIndex(start=0, stop=len(dataset_copy))

            return dataset_copy

    @staticmethod
    def convert_dataframe_to_timeseries(
        dataset: pandas.DataFrame, static_dataset: Optional[pandas.DataFrame] = None
    ):
        """Convert dataset for compatibility with ``darts``.

        Parameters
        ----------
        dataset : pandas.DataFrame
            dataset to convert from
        static_dataset : pandas.DataFrame, optional (default=None)
            source dataset for constant covariates to convert from

        Returns
        -------
        darts.TimeSeries
            converted dataset
        """
        from darts import TimeSeries

        return TimeSeries.from_dataframe(dataset, static_covariates=static_dataset)

    def _fit(
        self: "DartsRegressionModel",
        y: pandas.DataFrame,
        X: Optional[pandas.DataFrame],
        fh: Optional[ForecastingHorizon],
    ) -> "DartsRegressionModel":
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : pandas.DataFrame
            Time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : pandas.DataFrame, optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : DartsRegressionModel
            reference to self
        """
        del fh  # avoid being detected as unused by ``vulture`` like tools

        if self.needs_X and (
            X is None
            or not set(X.columns).issuperset(self._all_covariates)
            or X[self._all_covariates].empty
        ):
            raise ValueError(
                f"Missing exogeneous data, expected: {self._all_covariates}."
            )

        if not self._all_covariates:
            future_unknown_history = None
            future_known_history = None

            endogenous_history = self.convert_dataframe_to_timeseries(
                self.convert_dataset_index(y)
            )
        else:
            future_unknown_history = (
                self.convert_dataframe_to_timeseries(
                    self.convert_dataset_index(X[self._past_covariates])
                )
                if self._past_covariates
                else None
            )
            future_known_history = (
                self.convert_dataframe_to_timeseries(
                    self.convert_dataset_index(X[self._future_covariates])
                )
                if self._future_covariates
                else None
            )

            endogenous_history = self.convert_dataframe_to_timeseries(
                self.convert_dataset_index(y),
                static_dataset=(
                    self.convert_dataset_index(X[self._static_covariates])
                    if self._static_covariates
                    else None
                ),
            )

        self._forecaster = self.create_forecaster()

        _ = self._forecaster.fit(
            endogenous_history,
            past_covariates=future_unknown_history,
            future_covariates=future_known_history,
            max_samples_per_ts=self.max_samples_per_ts,
            n_jobs_multioutput_wrapper=self.n_jobs_multioutput_wrapper,
            **self._fit_kwargs,
        )

        return self

    def _predict(
        self: "DartsRegressionModel",
        fh: ForecastingHorizon,
        X: Optional[pandas.DataFrame],
    ) -> pandas.DataFrame:
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : pandas.DataFrame, optional (default=None)
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : pandas.DataFrame
            Point predictions
        """
        if not fh.is_all_out_of_sample(cutoff=self.cutoff):
            raise NotImplementedError("in-sample prediction is currently not supported")

        if self.needs_X and (
            X is None
            or not set(X.columns).issuperset(self._non_constant_covariates)
            or X[self._non_constant_covariates].empty
        ):
            raise ValueError(
                f"Missing exogeneous data, expected: {self._non_constant_covariates}."
            )

        if not self._non_constant_covariates:
            future_unknown_horizon = None
            future_known_horizon = None
        else:
            future_unknown_horizon = (
                self.convert_dataframe_to_timeseries(
                    self.convert_dataset_index(X[self._past_covariates])
                )
                if self._past_covariates
                else None
            )
            future_known_horizon = (
                self.convert_dataframe_to_timeseries(
                    self.convert_dataset_index(X[self._future_covariates])
                )
                if self._future_covariates
                else None
            )

        maximum_forecast_horizon = fh.to_relative(self.cutoff)[-1]

        model_forecasts = self._forecaster.predict(
            maximum_forecast_horizon,
            past_covariates=future_unknown_horizon,
            future_covariates=future_known_horizon,
            num_samples=self.num_samples,
            verbose=self.verbose,
            predict_likelihood_parameters=self.predict_likelihood_parameters,
            show_warnings=self.show_warnings,
            **self._predict_kwargs,
        )

        # TODO: extend to multi_models case
        model_point_predictions = model_forecasts.pd_dataframe()

        absolute_horizons = fh.to_absolute_index(self.cutoff)
        horizon_positions = fh.to_indexer(self.cutoff)

        final_point_predictions = model_point_predictions.iloc[horizon_positions]
        final_point_predictions.index = absolute_horizons
        final_point_predictions.columns = self._y.columns

        return final_point_predictions

    @classmethod
    def get_test_params(cls: Type["DartsRegressionModel"], parameter_set="default"):
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
        del parameter_set  # to avoid being detected as unused by `vulture` etc.

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        params = [
            {"lags": 2, "model": LinearRegression()},
            {"lags": [-1, -3], "model": RandomForestRegressor()},
        ]

        return params
