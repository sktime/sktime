# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Greykite forecaster for sktime."""

__author__ = ["vedantag17"]

import copy

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class GreykiteForecaster(BaseForecaster):
    """Adapter for using Greykite forecasting models within sktime.

    This forecaster wraps Greykite forecast_pipeline (configured via a ForecastConfig)
    and exposes a sktime-compatible API.

    WARNING: the ``greykite`` package has very restrictive dependencies that typically
    prevent installation together with other packages.

    Parameters
    ----------
    date_format : str, optional
        Format of the timestamp in the data. If None, it is inferred.
    model_template : str, optional
        Name of the model template to use (default: "SILVERKITE").
    coverage : float, optional
        Intended coverage of the prediction bands (0.0 to 1.0).
    cv_max_splits : int or None, optional
        Maximum number of cross-validation splits used by greykite's
        internal pipeline.  Set to ``0`` to skip CV entirely.
        If ``None``, greykite's default (3) is used.

    Attributes
    ----------
    _forecaster : object
        The fitted Greykite forecaster.
    _forecast : pandas.DataFrame
        The forecast result from the Greykite model.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.greykite import GreykiteForecaster
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> y = load_airline()
    >>> fh = ForecastingHorizon([1, 2, 3])
    >>> forecaster = GreykiteForecaster()
    >>> forecaster.fit(y=y, fh=fh)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=fh) # doctest: +SKIP

    References
    ----------
    .. [1] https://linkedin.github.io/greykite/docs/1.0.0/html/pages/stepbystep/0400_configuration.html

    """

    _tags = {
        # packaging info
        # --------------
        "python_dependencies": ["greykite>=1.0.0"],
        # estimator type
        # --------------
        "capability:multivariate": False,  # Handles univariate targets here.
        "capability:exogenous": True,
        "capability:missing_values": True,  # Handles missing data.
        "y_inner_mtype": "pd.Series",  # Expected input type for y.
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
            "test_deepcopy_fitted",
            "test_deepcopy_fitted_predict",
        ],
        "tests:python_dependencies": ["prophet", "setuptools<82"],
    }

    def __init__(
        self,
        date_format: str | None = None,
        model_template: str = "SILVERKITE",
        coverage: float = 0.95,
        cv_max_splits: int | None = None,
    ):
        self.date_format = date_format
        self.model_template = model_template
        self.coverage = coverage
        self.cv_max_splits = cv_max_splits

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

    def _create_forecast_config(self, y=None):
        """Create a ForecastConfig from the constructor parameters.

        The resolved config is stored in ``self._forecast_config_`` so that
        constructor hyperparameters are never mutated during ``fit``, as
        required by sktime's estimator contract.
        """
        # If frequency is not provided, try to infer it from the index.
        # pd.infer_freq only supports DatetimeIndex / PeriodIndex; for integer
        # or other index types we leave freq as None.
        if y is not None:
            if isinstance(y.index, pd.PeriodIndex):
                freq = pd.infer_freq(y.index.to_timestamp())
            elif isinstance(y.index, pd.DatetimeIndex):
                freq = pd.infer_freq(y.index)
            else:
                freq = None
        else:
            freq = None

        # Set train_end_date only for datetime-like indices; greykite cannot
        # parse an integer as a date.
        if y is not None and isinstance(y.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            train_end_date = y.index.max()
            if isinstance(y.index, pd.PeriodIndex):
                train_end_date = train_end_date.to_timestamp()
        else:
            train_end_date = None

        from greykite.framework.templates.autogen.forecast_config import (
            ComputationParam,
            EvaluationMetricParam,
            EvaluationPeriodParam,
            ForecastConfig,
            MetadataParam,
            ModelComponentsParam,
        )

        # Expects DataFrame with timestamp column named "ts" and value column named "y".
        metadata_param = MetadataParam(
            time_col="ts",
            value_col="y",
            date_format=self.date_format,
            freq=freq,
            train_end_date=train_end_date,
        )
        # Default model components.
        model_components_param = ModelComponentsParam()

        # Build the config and store it as a fitted attribute, NOT as a
        # hyperparameter, so that self.forecast_config stays None.
        self._forecast_config_ = ForecastConfig(
            metadata_param=metadata_param,
            model_components_param=model_components_param,
            model_template=self.model_template,
            coverage=self.coverage,
            evaluation_metric_param=EvaluationMetricParam(),
            evaluation_period_param=EvaluationPeriodParam(
                cv_max_splits=self.cv_max_splits,
            ),
            computation_param=ComputationParam(),
            forecast_one_by_one=False,
        )
        return self._forecast_config_

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

        # Preserve the series name so _predict can return a named Series.
        self._y_name_ = y.name

        self._y_train_ = y
        self._X_train_ = X
        self._X_cols_ = list(X.columns) if X is not None else []

        # Build the greykite DataFrame without mutating y.index.
        # IMPORTANT: never do y.index = y.index.to_timestamp() in-place.
        if isinstance(y.index, pd.PeriodIndex):
            ts_index = y.index.to_timestamp()
        else:
            ts_index = y.index
        df = pd.DataFrame({"ts": ts_index, "y": y.values})

        # Add exogenous columns to the training DataFrame.
        if X is not None:
            for col in X.columns:
                df[col] = X[col].values

        # Create the forecast configuration if not already provided.
        # Use a shallow copy so that setting forecast_horizon does not mutate
        # either self.forecast_config (the user-supplied hyperparameter) or
        # self._forecast_config_ across repeated fit calls.
        fc = copy.copy(self._create_forecast_config(y))
        # Convert fh to relative integer steps so that both relative fh
        # (e.g. [1, 2, 3]) and absolute fh (e.g. Period objects) work correctly.
        steps = np.array(fh.to_relative(self.cutoff).to_numpy(), dtype=int)
        fc.forecast_horizon = int(steps.max())

        # Fit the model using Greykite's forecast_pipeline.
        from greykite.framework.templates.forecaster import Forecaster

        result = Forecaster().run_forecast_config(df, fc)
        self._forecaster = result
        return self

    def _predict(self, fh=None, X=None):
        """Generate forecasts.

        Uses the stored results and returns predictions as a pandas Series.
        If exogenous variables were used in training and ``X`` is provided,
        re-runs the pipeline with future regressor values appended because
        greykite's ``forecast_pipeline`` requires future regressor values in
        the input DataFrame.
        """
        if fh is None:
            fh = self._fh

        # If exogenous columns were used in training and future X is provided,
        # re-run the pipeline so greykite can use the future regressor values.
        if self._X_cols_ and X is not None:
            # Rebuild training DataFrame.
            if isinstance(self._y_train_.index, pd.PeriodIndex):
                ts_index = self._y_train_.index.to_timestamp()
            else:
                ts_index = self._y_train_.index
            train_df = pd.DataFrame({"ts": ts_index, "y": self._y_train_.values})
            if self._X_train_ is not None:
                for col in self._X_train_.columns:
                    train_df[col] = self._X_train_[col].values

            # Build future rows: timestamps from fh, y=NaN, X columns filled.
            abs_idx = fh.to_absolute(self.cutoff).to_pandas()
            if isinstance(abs_idx, pd.PeriodIndex):
                abs_idx = abs_idx.to_timestamp()

            future_df = pd.DataFrame({"ts": abs_idx, "y": np.nan})
            for col in self._X_cols_:
                if col in X.columns:
                    future_df[col] = X[col].values

            full_df = pd.concat([train_df, future_df], ignore_index=True)

            fc = copy.copy(self._create_forecast_config(self._y_train_))
            steps = np.array(fh.to_relative(self.cutoff).to_numpy(), dtype=int)
            fc.forecast_horizon = int(steps.max())

            from greykite.framework.templates.forecaster import Forecaster

            forecaster = Forecaster().run_forecast_config(full_df, fc)
        else:
            forecaster = self._forecaster

        forecast_df = forecaster.forecast.df_test
        # Convert fh to relative integer steps (handles both relative and
        # Period-based absolute fh) then map to 0-based positions.
        steps = np.array(fh.to_relative(self.cutoff).to_numpy(), dtype=int)
        positions = (steps - 1).astype(int)

        preds = forecast_df["forecast"].values
        selected_preds = preds[positions]

        # Use sktime's authoritative absolute forecast dates as the index rather
        # than greykite's internal timestamps, so the result is consistent with
        # what sktime (and callers like test_hierarchical_with_exogeneous) expect.
        abs_idx = fh.to_absolute(self.cutoff).to_pandas()

        y_pred = pd.Series(selected_preds, index=abs_idx, name=self._y_name_)
        return y_pred

    def get_fitted_params(self):
        """Return fitted parameters."""
        self.check_is_fitted()
        if self._forecaster is None:
            # Vectorized (multivariate) case: sktime called _fit on per-variable
            # clones rather than on this outer instance, so _forecaster was never
            # set here.  Return what is available on the outer instance.
            return {"forecast_config": getattr(self, "_forecast_config_", None)}
        return {
            "model": self._forecaster.model,
            "forecast_config": self._forecast_config_,
        }

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
        params : list of dict
            Each dictionary contains parameters to construct a valid test
            instance of GreykiteForecaster.
        """
        # SILVERKITE_EMPTY: no hyperparameter grid search.
        # cv_max_splits=0: skips the internal CV loop (default is 3 splits,
        #   each 2+ GB on airline data) to prevent OOM kills on CI runners.
        return [
            {
                "model_template": "SILVERKITE_EMPTY",
                "cv_max_splits": 1,
            },
            {
                "model_template": "SILVERKITE_EMPTY",
                "cv_max_splits": 1,
                "coverage": 0.9,
            },
        ]
