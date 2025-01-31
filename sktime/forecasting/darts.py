# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from darts by Unit8."""

from typing import Optional

from sktime.forecasting.base.adapters._darts import (
    FUTURE_LAGS_TYPE,
    LAGS_TYPE,
    PAST_LAGS_TYPE,
    _DartsMixedCovariatesTorchModelAdapter,
    _DartsRegressionAdapter,
    _DartsRegressionModelsAdapter,
)
from sktime.utils.warnings import warn

__author__ = ["yarnabrina", "fnhirwa"]


class DartsRegressionModel(_DartsRegressionAdapter):
    """Darts Regression Model Estimator.

    Parameters
    ----------
    lags : One of int, list, dict, default=None
        Lagged target values used to predict the next time step. If an integer is given
        the last `lags` past lags are used (from -1 backward). Otherwise a list of
        integers with lags is required (each lag must be < 0). If a dictionary is given,
        keys correspond to the component names
        (of first series when using multiple series) and
        the values correspond to the component lags(integer or list of integers).
    lags_past_covariates : One of int, list, dict, default=None
        Number of lagged past_covariates values used to predict the next time step. If
        an integer is given the last `lags_past_covariates` past lags are used
        (inclusive, starting from lag -1). Otherwise a list of integers
        with lags < 0 is required. If a dictionary is given, keys correspond to the
        past_covariates component names(of first series when using multiple series)
        and the values correspond to the component lags(integer or list of integers).
    lags_future_covariates : One of tuple, list, dict, default=None
        Number of lagged future_covariates values used to predict the next time step. If
        a tuple (past, future) is given the last `past` lags in the past are used
        (inclusive, starting from lag -1) along with the first `future` future lags
        (starting from 0 - the prediction time - up to `future - 1` included). Otherwise
        a list of integers with lags is required. If dictionary is given,
        keys correspond to the future_covariates component names
        (of first series when using multiple series) and the values
        correspond to the component lags(integer or list of integers).
    output_chunk_shift : int, default=0
        Optionally, the number of steps to shift the start of the output chunk into the
        future (relative to the input chunk end). This will create a gap between the
        input (history of target and past covariates) and output. If the model supports
        future_covariates, the lags_future_covariates are relative to the first step in
        the shifted output chunk. Predictions will start output_chunk_shift steps after
        the end of the target series. If output_chunk_shift is set, the model cannot
        generate autoregressive predictions (n > output_chunk_length).
    output_chunk_length : int, default=1
        Number of time steps predicted at once by the internal regression model. Does
        not have to equal the forecast horizon `n` used in `predict()`. However, setting
        `output_chunk_length` equal to the forecast horizon may be useful if the
        covariates don't extend far enough into the future.
    add_encoders : dict, default=None
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
    model: object, default=None
        Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible
        to use model that doesn't support multi-output regression for multivariate
        timeseries, in which case one regressor will be used per component in the
        multivariate series. If None, defaults to:
        ``sklearn.linear_model.LinearRegression(n_jobs=-1)``.
    multi_models : bool, default=True
        If True, a separate model will be trained for each future lag to predict. If
        False, a single model is trained to predict at step 'output_chunk_length' in the
        future. Default: True.
    use_static_covariates : bool, default=True
        Whether the model should use static covariate information in case the input
        `series` passed to ``fit()`` contain static covariates. If ``True``, and static
        covariates are available at fitting time, will enforce that all target `series`
        have the same static covariate dimensionality in ``fit()`` and ``predict()``.

    past_covariates : list, default=None
        column names in ``X`` which are known only for historical data, by default None
    num_samples : int, default=1000
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
        # estimator type
        # --------------
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
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
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            add_encoders=add_encoders,
            multi_models=multi_models,
            model=model,
            use_static_covariates=use_static_covariates,
            past_covariates=past_covariates,
            num_samples=num_samples,
        )

    def _create_forecaster(self: "DartsRegressionModel"):
        """Create Darts model."""
        from darts.models import RegressionModel
        from sklearn.base import clone
        from sklearn.linear_model import LinearRegression

        if self.model is None:
            self._model = LinearRegression(n_jobs=-1)
        else:
            self._model = clone(self.model)

        model = RegressionModel(
            lags=self.lags,
            lags_past_covariates=self.lags_past_covariates,
            lags_future_covariates=self.lags_future_covariates,
            output_chunk_length=self.output_chunk_length,
            output_chunk_shift=self.output_chunk_shift,
            add_encoders=self.add_encoders,
            model=self._model,
            multi_models=self.multi_models,
            use_static_covariates=self.use_static_covariates,
        )
        return model

    @classmethod
    def get_test_params(cls: "DartsRegressionModel", parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

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
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        params = [
            {
                "num_samples": 100,
                "lags": 2,
                "add_encoders": None,
                "multi_models": False,
                "use_static_covariates": True,
            },
            {"lags": [-1, -3], "model": RandomForestRegressor()},
            {"lags": 3, "model": LinearRegression()},
        ]

        return params


class DartsXGBModel(_DartsRegressionModelsAdapter):
    """Darts XGBModel Estimator.

    This is based on implementation of XGBoost Model in darts [1]_ by Unit8.

    Parameters
    ----------
    lags : One of int, list, dict, default=None
        Lagged target values used to predict the next time step. If an integer is given
        the last `lags` past lags are used (from -1 backward). Otherwise a list of
        integers with lags is required (each lag must be < 0). If a dictionary is given,
        keys correspond to the component names
        (of first series when using multiple series) and
        the values correspond to the component lags(integer or list of integers).
    lags_past_covariates : One of int, list, dict, default=None
        Number of lagged past_covariates values used to predict the next time step. If
        an integer is given the last `lags_past_covariates` past lags are used
        (inclusive, starting from lag -1). Otherwise a list of integers
        with lags < 0 is required. If a dictionary is given, keys correspond to the
        past_covariates component names(of first series when using multiple series)
        and the values correspond to the component lags(integer or list of integers).
    lags_future_covariates : One of tuple, list, dict, default=None
        Number of lagged future_covariates values used to predict the next time step. If
        a tuple (past, future) is given the last `past` lags in the past are used
        (inclusive, starting from lag -1) along with the first `future` future lags
        (starting from 0 - the prediction time - up to `future - 1` included). Otherwise
        a list of integers with lags is required. If dictionary is given,
        keys correspond to the future_covariates component names
        (of first series when using multiple series) and the values
        correspond to the component lags(integer or list of integers).
    output_chunk_length : int, default=1
        Number of time steps predicted at once by the internal regression model. Does
        not have to equal the forecast horizon `n` used in `predict()`. However, setting
        `output_chunk_length` equal to the forecast horizon may be useful if the
        covariates don't extend far enough into the future.
    add_encoders : dict, default=None
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
    likelihood : str, default=None
        Can be set to `poisson` or `quantile`. If set, the model will be probabilistic,
        allowing sampling at prediction time. This will overwrite any `objective`
        parameter.
    quantiles : list, default=None
        Fit the model to these quantiles if the `likelihood` is set to `quantile`.
    random_state : int, default=None
        Control the randomness in the fitting procedure and for sampling. Default:
        ``None``.
    multi_models : bool, default=True
        If True, a separate model will be trained for each future lag to predict. If
        False, a single model is trained to predict at step 'output_chunk_length' in the
        future. Default: True.
    use_static_covariates : bool, default=True
        Whether the model should use static covariate information in case the input
        `series` passed to ``fit()`` contain static covariates. If ``True``, and static
        covariates are available at fitting time, will enforce that all target `series`
        have the same static covariate dimensionality in ``fit()`` and ``predict()``.
    past_covariates : list, default=None
        column names in ``X`` which are known only for historical data, by default None
    num_samples : int, default=1000
        Number of times a prediction is sampled from a probabilistic model, by default
        1000
    kwargs : dict, default=None
        Additional keyword arguments passed to `xgb.XGBRegressor`.
        Passed as a dictionary to conform to sklearn's API. Default: ``None``.

    References
    ----------
    .. [1] https://github.com/unit8co/darts

    Notes
    -----
    If unspecified, all columns will be assumed to be known during prediction duration.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["yarnabrina", "fnhirwa"],
        "maintainers": ["yarnabrina", "fnhirwa"],
        # estimator type
        # --------------
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:insample": False,
    }

    def __init__(
        self: "DartsXGBModel",
        past_covariates: Optional[list[str]] = None,
        num_samples: Optional[int] = 1000,
        lags: LAGS_TYPE = None,
        lags_past_covariates: PAST_LAGS_TYPE = None,
        lags_future_covariates: FUTURE_LAGS_TYPE = None,
        output_chunk_length: Optional[int] = 1,
        add_encoders: Optional[dict] = None,
        likelihood: Optional[str] = None,
        quantiles: Optional[list[float]] = None,
        random_state: Optional[int] = None,
        multi_models: Optional[bool] = True,
        use_static_covariates: Optional[bool] = True,
        kwargs: Optional[dict] = None,
    ) -> None:
        self.lags = lags
        self.lags_past_covariates = lags_past_covariates
        self.lags_future_covariates = lags_future_covariates
        self.output_chunk_length = output_chunk_length
        self.add_encoders = add_encoders
        self.likelihood = likelihood
        self.quantiles = quantiles
        self.random_state = random_state
        self.multi_models = multi_models
        self.use_static_covariates = use_static_covariates
        self.kwargs = kwargs

        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            add_encoders=add_encoders,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            past_covariates=past_covariates,
            num_samples=num_samples,
        )

    def _create_forecaster(self: "DartsXGBModel"):
        """Create Darts model."""
        from darts.models import XGBModel

        kwargs = self.kwargs or {}
        if self.quantiles is not None and self.multi_models:
            warn(
                (
                    "Setting multi_models=True with quantile regression may"
                    " cause issues. Consider using multi_models=False."
                ),
                obj=self,
                stacklevel=2,
            )
        return XGBModel(
            lags=self.lags,
            lags_past_covariates=self.lags_past_covariates,
            lags_future_covariates=self.lags_future_covariates,
            output_chunk_length=self.output_chunk_length,
            add_encoders=self.add_encoders,
            likelihood=self.likelihood,
            quantiles=self.quantiles,
            random_state=self.random_state,
            multi_models=self.multi_models,
            use_static_covariates=self.use_static_covariates,
            **kwargs,
        )

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
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        params = [
            {
                "num_samples": 100,
                "lags": 12,
                "output_chunk_length": 1,
                "add_encoders": None,
                "likelihood": "quantile",
                "quantiles": None,
                "random_state": None,
                "multi_models": False,
                "use_static_covariates": True,
                "kwargs": {
                    "objective": "reg:squarederror",
                },
            },
            {
                "num_samples": 200,
                "lags": 12,
                "output_chunk_length": 3,
                "add_encoders": None,
                "likelihood": "poisson",
                "quantiles": None,
                "random_state": None,
                "multi_models": False,
                "use_static_covariates": True,
                "kwargs": {"objective": "reg:squarederror"},
            },
        ]
        return params


class DartsLinearRegressionModel(_DartsRegressionModelsAdapter):
    """Darts LinearRegression Estimator.

    This is based on implementation of Regression Model in darts [1]_ by Unit8.

    Parameters
    ----------
    lags : One of int, list, dict, default=None
        Lagged target values used to predict the next time step. If an integer is given
        the last `lags` past lags are used (from -1 backward). Otherwise a list of
        integers with lags is required (each lag must be < 0). If a dictionary is given,
        keys correspond to the component names
        (of first series when using multiple series) and
        the values correspond to the component lags(integer or list of integers).
    lags_past_covariates : One of int, list, dict, default=None
        Number of lagged past_covariates values used to predict the next time step. If
        an integer is given the last `lags_past_covariates` past lags are used
        (inclusive, starting from lag -1). Otherwise a list of integers
        with lags < 0 is required. If a dictionary is given, keys correspond to the
        past_covariates component names(of first series when using multiple series)
        and the values correspond to the component lags(integer or list of integers).
    lags_future_covariates : One of tuple, list, dict, default=None
        Number of lagged future_covariates values used to predict the next time step. If
        a tuple (past, future) is given the last `past` lags in the past are used
        (inclusive, starting from lag -1) along with the first `future` future lags
        (starting from 0 - the prediction time - up to `future - 1` included). Otherwise
        a list of integers with lags is required. If dictionary is given,
        keys correspond to the future_covariates component names
        (of first series when using multiple series) and the values
        correspond to the component lags(integer or list of integers).
    output_chunk_length : int, default=1
        Number of time steps predicted at once by the internal regression model. Does
        not have to equal the forecast horizon `n` used in `predict()`. However, setting
        `output_chunk_length` equal to the forecast horizon may be useful if the
        covariates don't extend far enough into the future.
    add_encoders : dict, default=None
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
    likelihood : str, default=None
        Can be set to `poisson` or `quantile`. If set, the model will be probabilistic,
        allowing sampling at prediction time. This will overwrite any `objective`
        parameter.
    quantiles : list, default=None
        Fit the model to these quantiles if the `likelihood` is set to `quantile`.
    random_state : int, default=None
        Control the randomness in the fitting procedure and for sampling. Default:
        ``None``.
    multi_models : bool, default=True
        If True, a separate model will be trained for each future lag to predict. If
        False, a single model is trained to predict at step 'output_chunk_length' in the
        future. Default: True.
    use_static_covariates: bool, default=True
        Whether the model should use static covariate information in case the input
        `series` passed to ``fit()`` contain static covariates. If ``True``, and static
        covariates are available at fitting time, will enforce that all target `series`
        have the same static covariate dimensionality in ``fit()`` and ``predict()``.
    past_covariates : list, default=None
        column names in ``X`` which are known only for historical data, by default None
    num_samples : int, default=1000
        Number of times a prediction is sampled from a probabilistic model, by default
        1000
    kwargs
        Additional keyword arguments passed to `sklearn.linear_model.LinearRegression`
        (by default), to `sklearn.linear_model.PoissonRegressor`
        if `likelihood` is `poisson`,
        or to `sklearn.linear_model.QuantileRegressor` if `likelihood` is `quantile`.
        Passed as a dictionary to conform to sklearn's API. Default: ``None``.

    References
    ----------
    .. [1] https://github.com/unit8co/darts

    Notes
    -----
    If unspecified, all columns will be assumed to be known during prediction duration.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fnhirwa"],
        "maintainers": ["fnhirwa"],
        # estimator type
        # --------------
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:insample": False,
    }

    def __init__(
        self: "DartsLinearRegressionModel",
        past_covariates: Optional[list[str]] = None,
        num_samples: Optional[int] = 1000,
        lags: LAGS_TYPE = None,
        lags_past_covariates: PAST_LAGS_TYPE = None,
        lags_future_covariates: FUTURE_LAGS_TYPE = None,
        output_chunk_length: Optional[int] = 1,
        add_encoders: Optional[dict] = None,
        likelihood: Optional[str] = None,
        quantiles: Optional[list[float]] = None,
        random_state: Optional[int] = None,
        multi_models: Optional[bool] = True,
        use_static_covariates: Optional[bool] = True,
        kwargs: Optional[dict] = None,
    ) -> None:
        self.lags = lags
        self.lags_past_covariates = lags_past_covariates
        self.lags_future_covariates = lags_future_covariates
        self.output_chunk_length = output_chunk_length
        self.add_encoders = add_encoders
        self.likelihood = likelihood
        self.quantiles = quantiles
        self.random_state = random_state
        self.multi_models = multi_models
        self.use_static_covariates = use_static_covariates
        self.kwargs = kwargs

        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            add_encoders=add_encoders,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            past_covariates=past_covariates,
            num_samples=num_samples,
        )

    def _create_forecaster(self: "DartsLinearRegressionModel"):
        """Create Darts model."""
        from darts.models import LinearRegressionModel

        kwargs = self.kwargs or {}
        if self.quantiles is not None and self.multi_models:
            warn(
                (
                    "Setting multi_models=True with quantile regression may"
                    " cause issues. Consider using multi_models=False."
                ),
                obj=self,
                stacklevel=2,
            )
        return LinearRegressionModel(
            lags=self.lags,
            lags_past_covariates=self.lags_past_covariates,
            lags_future_covariates=self.lags_future_covariates,
            output_chunk_length=self.output_chunk_length,
            add_encoders=self.add_encoders,
            likelihood=self.likelihood,
            quantiles=self.quantiles,
            random_state=self.random_state,
            multi_models=self.multi_models,
            use_static_covariates=self.use_static_covariates,
            **kwargs,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

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
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        params = [
            {
                "num_samples": 100,
                "lags": 12,
                "output_chunk_length": 1,
                "add_encoders": None,
                "likelihood": "quantile",
                "quantiles": None,
                "random_state": None,
                "multi_models": False,
                "use_static_covariates": True,
                "kwargs": {
                    "fit_intercept": True,
                },
            },
            {
                "num_samples": 200,
                "lags": 12,
                "output_chunk_length": 3,
                "add_encoders": None,
                "likelihood": "poisson",
                "quantiles": None,
                "random_state": None,
                "multi_models": False,
                "use_static_covariates": True,
                "kwargs": None,
            },
        ]

        return params


class DartsTiDEModel(_DartsMixedCovariatesTorchModelAdapter):
    """TiDE (Time-series Dense Encoders) Forecaster.

    This is based on the implementation of the TiDE Model in darts

    Parameters
    ----------
    input_chunk_length : int
        Number of time steps in the past to take as a model input (per chunk). Applies
        to the target series, and past and/or future covariates.
    output_chunk_length : int
        Number of time steps predicted at once (per chunk) by the internal model. Also,
        the number of future values from future covariates to use as a model input
        (if the model supports future covariates). It is not the same as forecast
        horizon n used in predict(), which is the desired number of prediction points
        generated using either a one-shot- or autoregressive forecast. Setting
        n <= output_chunk_length prevents auto-regression. This is useful when the
        covariates do not extend far enough into the future, or to prohibit the model
        from using future values of past and / or future covariates for prediction
        (depending on the model covariate support).
    output_chunk_shift : int, optional (default=0)
        Optionally, the number of steps to shift the start of the output chunk into the
        future (relative to the input chunk end). This will create a gap between the
        input and output. If the model supports future_covariates, the future values are
        extracted from the shifted output chunk. Predictions will start
        output_chunk_shift steps after the end of the target series. If
        output_chunk_shift is set, the model cannot generate autoregressive predictions
        (n > output_chunk_length).
    num_encoder_layers : int, optional (default=1)
        The number of residual blocks in the encoder.
    num_decoder_layers : int, optional (default=1)
        The number of residual blocks in the decoder.
    decoder_output_dim : int, optional (default=16)
        Dimension of decoder output.
    hidden_size : int, optional (default=128)
        The width of the layers in the residual blocks between the encoder and decoder.
    temporal_width_past : int, optional (default=4)
        The width of the output layer in the past covariate projection residual block.
        If 0, will bypass feature projection and use the raw feature data.
    temporal_width_future : int, optional (default=4)
        The width of the output layer in the future covariate projection residual block.
        If 0, will bypass feature projection and use the raw feature data.
    temporal_decoder_hidden : int, optional (default=32)
        The width of the layers in the temporal decoder.
    use_layer_norm : bool, optional (default=False)
        Whether to use layer normalization
    dropout : float, optional (default=0.1)
        Dropout probability to be used in fully connected layers.
    use_static_covariates : bool, optional (default=True)
        Whether to use static covariates
    kwargs: dict, optional (default=True`)
        Optional arguments to initialize the pytorch_lightning.Module,
        pytorch_lightning.Trainer.

    References
    ----------
    .. [1] https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tide_model.html

    Notes
    -----
    If unspecified, all columns will be assumed to be known during predictions duration.
    """

    _tags = {
        "authors": ["PranavBhatP"],
        "maintainers": ["PranavBhatP"],
        "python_version": ">=3.9",
        "python_dependencies": ["darts>=0.29"],
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "capability:insample": True,
        "capability:pred_int": True,
    }

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        decoder_output_dim: int = 16,
        hidden_size: int = 128,
        temporal_width_past: int = 4,
        temporal_width_future: int = 4,
        temporal_decoder_hidden: int = 32,
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        use_static_covariates: bool = True,
        kwargs: Optional[dict] = None,
    ):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.output_chunk_shift = output_chunk_shift
        self.use_static_covariates = use_static_covariates
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_width_past = temporal_width_past
        self.temporal_width_future = temporal_width_future
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.kwargs = kwargs

        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            use_static_covariates=use_static_covariates,
        )

    def _create_forecaster(self: "DartsTiDEModel"):
        """Create and initialize a TiDE forecaster instance."""
        from darts.models import TiDEModel

        kwargs = self.kwargs or {}
        return TiDEModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            output_chunk_shift=self.output_chunk_shift,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_output_dim=self.decoder_output_dim,
            hidden_size=self.hidden_size,
            temporal_width_past=self.temporal_width_past,
            temporal_width_future=self.temporal_width_future,
            temporal_decoder_hidden=self.temporal_decoder_hidden,
            use_layer_norm=self.use_layer_norm,
            dropout=self.dropout,
            use_static_covariates=self.use_static_covariates,
            **kwargs,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameters settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default = "default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, returns `"default"` set.

        Returns
        -------
        params: dict or list of dict
        """
        del parameter_set

        params = [
            {
                "input_chunk_length": 12,
                "output_chunk_length": 6,
            },
            {
                "input_chunk_length": 24,
                "output_chunk_length": 12,
                "hidden_size": 64,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "temporal_width_past": 8,
                "temporal_width_future": 8,
                "use_layer_norm": True,
                "dropout": 0.2,
            },
        ]
        return params


__all__ = [
    "DartsRegressionModel",
    "DartsXGBModel",
    "DartsLinearRegressionModel",
    "DartsTiDEModel",
]
