# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Greykite forecaster for sktime."""

__author__ = ["vedantag17"]

from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class GreykiteForecaster(BaseForecaster):
    """Adapter for using Greykite forecasting models within sktime.

    This forecaster wraps Greykite ``forecast_pipeline`` and exposes a
    sktime-compatible API.

    WARNING: the ``greykite`` package has very restrictive dependencies that typically
    prevent installation together with other packages. For this reason, this estimator
    is also not covered by regular tests. We therefore recommend to run
    ``check_estimator(GreykiteForecaster)`` on your system before deploying
    this estimator.

    Notes
    -----
    - Greykite ``ForecastConfig`` fields are available either via a ``forecast_config``
      object or as flattened constructor parameters.
    - If ``forecast_config`` is provided, it is used as the base configuration.
    - Flattened parameters whose default is ``None`` override the corresponding
      fields on that object when set.
    - ``model_template`` and ``coverage`` are applied only when ``forecast_config`` is
      ``None``, so their defaults do not overwrite values already set on a
      user-provided config.
    - To change those fields when passing ``forecast_config``, set them on the config
      object itself.
    - The time index of ``y`` and ``X`` can be any format recognized by
      ``pandas.to_datetime``. If conversion fails, a default daily
      ``DatetimeIndex`` starting at ``2000-01-01`` is created for Greykite
      internally.

    Parameters
    ----------
    forecast_config : greykite ForecastConfig or None, default=None
        Optional Greykite ``ForecastConfig`` used as the base configuration.
        If None, a config is built from the flattened parameters below.
        Prefer the flattened parameters for discoverability; use
        ``forecast_config`` for advanced setups or when migrating existing
        Greykite code.
    date_format : str or None, default=None
        Format string for parsing timestamps (e.g. ``"%Y-%m-%d"``).
        If None, Greykite infers the format.
    model_template : str, default="SILVERKITE"
        Greykite model template name. Used only when ``forecast_config`` is
        None; otherwise set ``forecast_config.model_template`` instead.

        Main templates:

        * ``"SILVERKITE"`` - default Silverkite model with automatic growth,
          seasonality, holidays, autoregression, and interactions. Good
          general choice for hourly and daily data.
        * ``"PROPHET"`` - Facebook Prophet with growth, seasonality,
          holidays, regressors, and prediction intervals.
        * ``"AUTO_ARIMA"`` - ARIMA with automatic order selection.
        * ``"AUTO"`` - automatically selects a SimpleSilverkite template
          from data frequency, forecast horizon, and CV settings.
        * ``"SILVERKITE_EMPTY"`` - intercept-only Silverkite; add only the
          components you need via ``model_components_param``.
        * ``"SK"`` - low-level Silverkite interface for custom tuning; not
          intended out-of-the-box.
        * ``"LAG_BASED"`` - forecasts from aggregated past values
          (e.g. week-over-week).
        * ``"SILVERKITE_TWO_STAGE"`` / ``"SILVERKITE_WOW"`` - multistage
          models (long-term effects then short-term residuals, or
          Silverkite plus week-over-week).

        Frequency / horizon variants of Silverkite also exist, for example
        ``SILVERKITE_DAILY_1``, ``SILVERKITE_DAILY_90``, ``SILVERKITE_WEEKLY``,
        ``SILVERKITE_MONTHLY``, and ``SILVERKITE_HOURLY_{1,24,168,336}``.
        These use hyperparameters tuned for that frequency and horizon.
        See greykite ``ModelTemplateEnum`` for the full list [2]_.

    coverage : float, default=0.95
        Intended coverage of prediction bands, between 0 and 1. Used only
        when ``forecast_config`` is None; otherwise set
        ``forecast_config.coverage`` instead. If None on the config,
        Greykite does not return upper/lower prediction bounds.
    forecast_one_by_one : bool, int, list of int, or None, default=None
        If set, enables Greykite's one-by-one forecasting (fit/predict in
        segments of the horizon). ``True`` uses the full horizon as one
        segment; an int is a segment size; a list of ints must sum to the
        horizon. If None and ``forecast_config`` is None, defaults to
        False; if ``forecast_config`` is provided, that config's value is
        kept.
    freq : str or None, default=None
        Pandas frequency string for the series (e.g. ``"D"``, ``"H"``).
        If None, inferred from the index of ``y`` when possible. Maps to
        ``MetadataParam.freq``.
    anomaly_info : dict, list of dict, or None, default=None
        Optional anomaly adjustment specification. Values flagged here are
        corrected before fitting. Maps to ``MetadataParam.anomaly_info``.
    model_components_param : dict, ModelComponentsParam, list, or None, default=None
        Model structure and tuning options passed to Greykite
        ``ModelComponentsParam``. If None, template defaults are used.
        When a dict (or list of dicts for grid search), recognized keys are:

        * ``growth`` - trend / growth terms
        * ``seasonality`` - seasonal Fourier or dummy terms
        * ``events`` - holidays and other event effects
        * ``changepoints`` - trend changepoint placement and strength
        * ``autoregression`` - lagged value terms
        * ``regressors`` - contemporaneous exogenous regressors
        * ``lagged_regressors`` - lagged exogenous regressors
        * ``uncertainty`` - prediction-interval / uncertainty model
        * ``custom`` - template-specific extra options
        * ``hyperparameter_override`` - dict (or list of dicts) applied on
          top of the template hyperparameter grid for full customization

    evaluation_metric_param : dict, EvaluationMetricParam, or None, default=None
        Metrics used for CV reporting and model selection. Passed to
        Greykite ``EvaluationMetricParam``. When a dict, recognized keys are:

        * ``cv_selection_metric`` - metric name used to pick the best CV model
        * ``cv_report_metrics`` - extra metric name(s) reported during CV
        * ``agg_periods`` / ``agg_func`` - optional aggregation of the series
          before computing the metric (period count and aggregation function)
        * ``null_model_params`` - configuration for Greykite's null model
          baseline comparison
        * ``relative_error_tolerance`` - relative error threshold used by
          some Greykite diagnostics

    evaluation_period_param : dict, EvaluationPeriodParam, or None, default=None
        Train/test and cross-validation split configuration. Passed to
        Greykite ``EvaluationPeriodParam``. When a dict, recognized keys are:

        * ``test_horizon`` - holdout length at the end of the series
        * ``periods_between_train_test`` - gap between train and test
        * ``cv_horizon`` - forecast horizon used inside each CV fold
        * ``cv_max_splits`` - maximum number of CV splits
        * ``cv_min_train_periods`` - minimum training length per split
        * ``cv_periods_between_splits`` - step size between CV split starts
        * ``cv_periods_between_train_test`` - gap between CV train and test
        * ``cv_expanding_window`` - if True, use expanding rather than
          sliding training windows
        * ``cv_use_most_recent_splits`` - if True, prefer recent splits when
          capping ``cv_max_splits``

    computation_param : dict, ComputationParam, or None, default=None
        Runtime / parallelization options. Passed to Greykite
        ``ComputationParam``. When a dict, recognized keys are:

        * ``n_jobs`` - parallel jobs for hyperparameter search (``-1`` uses
          all processors)
        * ``hyperparameter_budget`` - max hyperparameter combinations to
          evaluate (None means the full grid)
        * ``verbose`` - verbosity level for fitting and CV logs

    Attributes
    ----------
    _forecaster : object
        The fitted Greykite forecaster.
    _forecast : pandas.DataFrame
        The forecast result from the Greykite model.
    _X : pandas.DataFrame
        The exogenous variables, if provided.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.greykite import GreykiteForecaster
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> y = load_airline().to_timestamp()
    >>> fh = ForecastingHorizon([1, 2, 3])
    >>> forecaster = GreykiteForecaster()
    >>> forecaster.fit(y=y, fh=fh)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=fh) # doctest: +SKIP

    References
    ----------
    .. [1] https://linkedin.github.io/greykite/docs/1.0.0/html/pages/stepbystep/0400_configuration.html
    .. [2] https://linkedin.github.io/greykite/docs/0.1.0/html/gallery/tutorials/0200_templates.html
    """

    _tags = {
        # packaging info
        # --------------
        "python_dependencies": ["greykite>=1.0.0"],
        # estimator type
        # --------------
        "capability:multivariate": False,  # Handles univariate targets here.
        "capability:exogenous": True,  # Can handle exogenous variables.
        "capability:missing_values": True,  # Handles missing data.
        "y_inner_mtype": "pd.Series",  # Expected input type for y.
        "X_inner_mtype": "pd.DataFrame",  # Expected input type for X.
        "requires-fh-in-fit": True,  # Forecasting horizon is required in fit.
        "capability:pred_int": False,  # Can produce prediction intervals.
        "capability:unequal_length": False,
        "capability:insample": False,
        # CI and test flags
        # -----------------
        "tests:vm": True,
        # greykite failures tracked in #10083
        "tests:skip_all": True,
        # pickling is not supported for GreykiteForecaster.
        # The greykite package internally uses patsy, which does not support
        # pickling or deepcopy (see https://github.com/pydata/patsy/issues/26).
        "tests:skip_by_name": [
            "test_fit_idempotent",
            "test_persistence_via_pickle",
            "test_save_estimators_to_file",
            "test_update_predict_predicted_index",
            "test_deepcopy_fitted_predict",
            "test_deepcopy_fitted",
        ],
        "tests:python_dependencies": ["prophet>1.2.1", "setuptools<82"],
    }

    def __init__(
        self,
        forecast_config: Optional["GreykiteForecaster.ForecastConfig"] = None,
        date_format: str | None = None,
        model_template: str = "SILVERKITE",
        coverage: float = 0.95,
        forecast_one_by_one=None,
        freq=None,
        anomaly_info=None,
        model_components_param=None,
        evaluation_metric_param=None,
        evaluation_period_param=None,
        computation_param=None,
    ):
        self.forecast_config = forecast_config
        self.date_format = date_format
        self.model_template = model_template
        self.coverage = coverage
        self.forecast_one_by_one = forecast_one_by_one
        self.freq = freq
        self.anomaly_info = anomaly_info
        self.model_components_param = model_components_param
        self.evaluation_metric_param = evaluation_metric_param
        self.evaluation_period_param = evaluation_period_param
        self.computation_param = computation_param

        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        if self.model_template == "PROPHET":
            self.set_tags(**{"python_dependencies": ["greykite>=1.0.0", "prophet"]})

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor

        IMPORTANT: no significant compute or memory use should happen in __post_init__,
        memory and compute intensive operations should be in _fit, not __post_init__.
        """
        self._forecaster = None
        self._forecast = None
        self._X = None

    @staticmethod
    def _ensure_datetime_index(y):
        """Ensure ``y`` has a DatetimeIndex for Greykite.

        Tries ``pandas.to_datetime`` on the index; if that fails, falls back to a
        default daily ``DatetimeIndex`` starting at ``2000-01-01``.
        """
        if y is None or isinstance(y.index, pd.DatetimeIndex):
            return y
        y = y.copy()
        try:
            y.index = pd.to_datetime(y.index)
        except (TypeError, ValueError, OverflowError):
            y.index = pd.date_range("2000-01-01", periods=len(y), freq="D")
        return y

    @staticmethod
    def _coerce_param(param_cls, param):
        """Coerce a dict or param instance to a Greykite param dataclass."""
        if param is None:
            return None
        if isinstance(param, param_cls):
            return param
        if isinstance(param, dict):
            return param_cls.from_dict(param)
        if isinstance(param, list):
            return [
                None
                if p is None
                else p
                if isinstance(p, param_cls)
                else param_cls.from_dict(p)
                for p in param
            ]
        raise TypeError(
            f"Expected None, dict, list, or {param_cls.__name__}, got {type(param)}."
        )

    def _create_forecast_config(self, y=None):
        """Create a ForecastConfig from ``forecast_config`` and flattened params.

        ``forecast_config`` (if given) is the base. Parameters defaulting to
        ``None`` override base fields only when explicitly set. ``model_template``
        and ``coverage`` are written only when no base config is provided.
        """
        from greykite.framework.templates.autogen.forecast_config import (
            ComputationParam,
            EvaluationMetricParam,
            EvaluationPeriodParam,
            ForecastConfig,
            MetadataParam,
            ModelComponentsParam,
        )

        freq = self.freq
        if freq is None and y is not None:
            freq = pd.infer_freq(y.index)
        train_end_date = y.index.max() if y is not None else None

        if self.forecast_config is not None:
            fc = deepcopy(self.forecast_config)
        else:
            # model_template / coverage only applied when building a fresh config,
            # so their defaults cannot clobber a user-provided forecast_config.
            forecast_one_by_one = self.forecast_one_by_one
            if forecast_one_by_one is None:
                forecast_one_by_one = False
            fc = ForecastConfig(
                model_template=self.model_template,
                coverage=self.coverage,
                forecast_one_by_one=forecast_one_by_one,
                model_components_param=ModelComponentsParam(),
                evaluation_metric_param=EvaluationMetricParam(),
                evaluation_period_param=EvaluationPeriodParam(),
                computation_param=ComputationParam(),
            )

        # Overlay flattened params that default to None (explicit set only).
        if self.model_components_param is not None:
            fc.model_components_param = self._coerce_param(
                ModelComponentsParam, self.model_components_param
            )
        if self.evaluation_metric_param is not None:
            fc.evaluation_metric_param = self._coerce_param(
                EvaluationMetricParam, self.evaluation_metric_param
            )
        if self.evaluation_period_param is not None:
            fc.evaluation_period_param = self._coerce_param(
                EvaluationPeriodParam, self.evaluation_period_param
            )
        if self.computation_param is not None:
            fc.computation_param = self._coerce_param(
                ComputationParam, self.computation_param
            )
        if self.forecast_config is not None and self.forecast_one_by_one is not None:
            fc.forecast_one_by_one = self.forecast_one_by_one

        # Metadata: adapter always feeds columns "ts" / "y".
        if fc.metadata_param is None:
            fc.metadata_param = MetadataParam()
        fc.metadata_param.time_col = "ts"
        fc.metadata_param.value_col = "y"
        if self.date_format is not None or self.forecast_config is None:
            fc.metadata_param.date_format = self.date_format
        if self.freq is not None:
            fc.metadata_param.freq = self.freq
        elif fc.metadata_param.freq is None:
            fc.metadata_param.freq = freq
        if self.anomaly_info is not None:
            fc.metadata_param.anomaly_info = self.anomaly_info
        if train_end_date is not None and fc.metadata_param.train_end_date is None:
            fc.metadata_param.train_end_date = train_end_date

        return fc

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Converts the input series into a DataFrame with columns "ts" and "y"
        and then runs the forecast_pipeline using the ForecastConfig.
        """
        # Ensure fh (forecasting horizon) is provided.
        if fh is None:
            raise ValueError(
                "The forecasting horizon `fh` must be provided in the `fit` method."
            )

        y = self._ensure_datetime_index(y)
        X = self._ensure_datetime_index(X)

        # Convert y into a DataFrame with columns "ts" and "y".
        df = pd.DataFrame({"ts": y.index, "y": y.values})

        # If exogenous variables X are provided, merge them into the DataFrame.
        if X is not None:
            for col in X.columns:
                df[col] = X[col].values
            self._X = X.copy()

        fc = self._create_forecast_config(y)
        steps = fh.to_relative(self.cutoff).to_numpy()
        fc.forecast_horizon = int(np.max(steps))
        self._forecast_config = fc

        # Fit the model using Greykite's forecast_pipeline.
        from greykite.framework.templates.forecaster import Forecaster

        result = Forecaster().run_forecast_config(df, fc)
        self._forecaster = result
        return self

    def _predict(self, fh=None, X=None):
        """Generate forecasts.

        Uses the stored results and returns predictions as a pandas Series.
        """
        if fh is None:
            fh = self._fh
        forecast_df = self._forecaster.forecast.df_test
        steps = fh.to_relative(self.cutoff).to_numpy()
        positions = (steps - 1).astype(int)
        selected_preds = forecast_df["forecast"].values[positions]
        return pd.Series(
            selected_preds,
            index=fh.to_absolute_index(self.cutoff),
            name=self._y.name,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the GreykiteForecaster.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the test parameter set to return. This forecaster supports a
            single default parameter set.

        Returns
        -------
        params : dict
            A dictionary containing parameters to construct a valid test instance of
            the GreykiteForecaster. The dictionary includes:
                - model_template: str
                    Name of the model template to use (default is 'SILVERKITE').
                - date_format: str or None
                    Format of the time column (default is None, allowing inference).
        """
        return [
            {
                "model_template": "SILVERKITE",
                "date_format": None,
            },
            {
                "model_template": "SILVERKITE",
                "date_format": None,
                "coverage": 0.95,
            },
            {
                "model_template": "PROPHET",
                "date_format": "%Y-%m-%d",
                "forecast_config": None,
                "coverage": 0.75,
            },
        ]
