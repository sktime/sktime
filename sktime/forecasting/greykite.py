# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Greykite forecaster for sktime."""

__author__ = ["vedantag17"]

from typing import Optional

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class GreykiteForecaster(BaseForecaster):
    """Adapter for using Greykite forecasting models within sktime.

    This forecaster wraps Greykite forecast_pipeline (configured via a ForecastConfig)
    and exposes a sktime-compatible API.

    WARNING: the ``greykite`` package has very restrictive dependencies that typically
    prevent installation together with other packages. For this reason, this estimator
    is also not covered by regular tests. We therefore recommend to run
    ``check_estimator(GreykiteForecaster)`` on your system before deploying
    this estimator.

    Parameters
    ----------
    forecast_config : ForecastConfig, optional
        Configuration object for Greykite's forecasting pipeline. If None,
         a default configuration
        is created.
    date_format : str, optional
        Format of the timestamp in the data. If None, it is inferred.
    model_template : str, optional
        Name of the model template to use (default: "SILVERKITE").
    coverage : float, optional
        Intended coverage of the prediction bands (0.0 to 1.0).

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
        "scitype:y": "univariate",  # Handles univariate targets here.
        "ignores-exogeneous-X": False,  # Can handle exogenous variables.
        "handles-missing-data": True,  # Handles missing data.
        "y_inner_mtype": "pd.Series",  # Expected input type for y.
        "X_inner_mtype": "pd.DataFrame",  # Expected input type for X.
        "requires-fh-in-fit": True,  # Forecasting horizon is required in fit.
        "capability:pred_int": False,  # Can produce prediction intervals.
        "capability:pickle": False,
        "capability:in-sample": False,
        "python_dependencies": ["greykite>=1.0.0"],  # Required Python dependencies.
    }

    def __init__(
        self,
        forecast_config: Optional["GreykiteForecaster.ForecastConfig"] = None,
        date_format: Optional[str] = None,
        model_template: str = "SILVERKITE",
        coverage: float = 0.95,
    ):
        super().__init__()
        self.forecast_config = forecast_config
        self.date_format = date_format
        self.model_template = model_template
        self.coverage = coverage

        self._forecaster = None
        self._forecast = None
        self._X = None

    def _create_forecast_config(self, y=None):
        """Create a ForecastConfig object if one wasn't provided."""
        if self.forecast_config is not None:
            return self.forecast_config

        # If frequency is not provided, try to infer it from the index.
        if y is not None:
            if isinstance(y.index, pd.PeriodIndex):
                freq = y.index.freqstr
            else:
                freq = pd.infer_freq(y.index)
        else:
            freq = None

        # Set train_end_date explicitly using the maximum timestamp in y
        train_end_date = y.index.max() if y is not None else None

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

        # Create the ForecastConfig using Greykite's parameters.
        self.forecast_config = ForecastConfig(
            metadata_param=metadata_param,
            model_components_param=model_components_param,
            model_template=self.model_template,
            coverage=self.coverage,
            evaluation_metric_param=EvaluationMetricParam(),
            evaluation_period_param=EvaluationPeriodParam(),
            computation_param=ComputationParam(),
            forecast_one_by_one=False,
        )
        return self.forecast_config

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

        if isinstance(y.index, pd.PeriodIndex):
            y.index = y.index.to_timestamp()
        # Convert y into a DataFrame with columns "ts" and "y".
        df = pd.DataFrame({"ts": y.index, "y": y.values})

        # If exogenous variables X are provided, merge them into the DataFrame.
        if X is not None:
            for col in X.columns:
                df[col] = X[col].values
            self._X = X.copy()

        # Create the forecast configuration if not already provided.
        fc = self._create_forecast_config(y)
        if hasattr(fh, "to_numpy"):
            steps = fh.to_numpy()
        else:
            steps = np.array(list(fh), dtype=int)
        fc.forecast_horizon = int(steps.max())

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
        if hasattr(fh, "to_numpy"):
            steps = fh.to_numpy()
        else:
            steps = np.array(list(fh), dtype=int)
        # compute zero-based positions
        positions = (steps - 1).astype(int)

        time_col = self._forecaster.forecast.time_col
        times = forecast_df[time_col].values
        preds = forecast_df["forecast"].values
        selected_times = times[positions]
        selected_preds = preds[positions]

        y_pred = pd.Series(selected_preds, index=selected_times)
        return y_pred

    def get_fitted_params(self):
        """Return fitted parameters."""
        if self._forecaster is None:
            raise ValueError("Forecaster has not been fitted yet. Call 'fit' first.")
        return {
            "model": self._forecaster.model,
            "forecast_config": self.forecast_config,
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
