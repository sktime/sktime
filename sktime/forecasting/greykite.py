# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Greykite forecaster for sktime."""

__author__ = ["vedantag17"]


import pandas as pd

from sktime.forecasting.base import BaseForecaster


class GreykiteForecaster(BaseForecaster):
    """Adapter for using Greykite forecasting models within sktime.

    This forecaster wraps Greykite forecast_pipeline (configured via a ForecastConfig)
    and exposes a sktime-compatible API.

    Parameters
    ----------
    forecast_config : ForecastConfig, optional
        Configuration object for Greykite's forecasting pipeline. If None,
         a default configuration
        is created.
    date_format : str, optional
        Format of the timestamp in the data. If None, it is inferred.
    freq : str, optional
        Frequency of the time series data (e.g., 'D' for daily, 'M' for monthly).
    forecast_horizon : int, optional
        The number of periods to forecast. If None, it will be set via the
        forecasting horizon.
    model_template : str, optional
        Name of the model template to use (default: "SILVERKITE").

    Attributes
    ----------
    _forecaster : object
        The fitted Greykite forecaster.
    _forecast : pandas.DataFrame
        The forecast result from the Greykite model.
    _X : pandas.DataFrame
        The exogenous variables, if provided.
    """

    # Import dependencies inside the class definition
    from typing import Optional

    from greykite.framework.pipeline.pipeline import forecast_pipeline
    from greykite.framework.templates.autogen.forecast_config import (
        ComputationParam,
        EvaluationMetricParam,
        EvaluationPeriodParam,
        ForecastConfig,
        MetadataParam,
        ModelComponentsParam,
    )

    _tags = {
        "scitype:y": "univariate",  # Handles univariate targets here.
        "ignores-exogeneous-X": False,  # Can handle exogenous variables.
        "handles-missing-data": True,  # Handles missing data.
        "y_inner_mtype": "pd.Series",  # Expected input type for y.
        "X_inner_mtype": "pd.DataFrame",  # Expected input type for X.
        "requires-fh-in-fit": True,  # Forecasting horizon is required in fit.
        "capability:pred_int": True,  # Can produce prediction intervals.
    }

    def __init__(
        self,
        forecast_config: Optional["GreykiteForecaster.ForecastConfig"] = None,
        date_format: Optional[str] = None,
        freq: Optional[str] = None,
        forecast_horizon: Optional[int] = None,
        model_template: str = "SILVERKITE",
        coverage: float = 0.95,
    ):
        super().__init__()
        self.forecast_config = forecast_config
        self.date_format = date_format
        self.freq = freq
        self.forecast_horizon = forecast_horizon
        self.model_template = model_template
        self.coverage = coverage

        self._forecaster = None
        self._forecast = None
        self._X = None

    def _create_forecast_config(self, y=None):
        """Create a ForecastConfig object if one wasn't provided."""
        if self.forecast_config is not None:
            return self.forecast_config

        if y is not None and self.freq is None:
            self.freq = pd.infer_freq(y.index)

        # Set train_end_date explicitly using the maximum timestamp in y
        train_end_date = y.index.max() if y is not None else None

        # Expects DataFrame with timestamp column named "ts" and value column named "y".
        metadata_param = self.MetadataParam(
            time_col="ts",
            value_col="y",
            date_format=self.date_format,
            freq=self.freq,
            train_end_date=train_end_date,
        )
        # Default model components.
        model_components_param = self.ModelComponentsParam()

        # Create the ForecastConfig using Greykite's parameters.
        self.forecast_config = self.ForecastConfig(
            metadata_param=metadata_param,
            model_components_param=model_components_param,
            model_template=self.model_template,
            forecast_horizon=self.forecast_horizon,
            coverage=self.coverage,
            evaluation_metric_param=self.EvaluationMetricParam(),
            evaluation_period_param=self.EvaluationPeriodParam(),
            computation_param=self.ComputationParam(),
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
        self._fh = fh

        # Convert y into a DataFrame with columns "ts" and "y".
        df = pd.DataFrame({"ts": y.index, "y": y.values}).reset_index(drop=True)

        # If exogenous variables X are provided, merge them into the DataFrame.
        if X is not None:
            for col in X.columns:
                df[col] = X[col].values
            self._X = X.copy()

        # Create the forecast configuration if not already provided.
        fc = self._create_forecast_config(y)
        fc.forecast_horizon = len(self._fh)

        # Fit the model using Greykite's forecast_pipeline.
        from greykite.framework.templates.forecaster import Forecaster

        result = Forecaster().run_forecast_config(df, fc)
        self._forecaster = result
        return self

    def _predict(self, fh=None, X=None):
        """Generate forecasts.

        Uses the stored results and returns predictions as a pandas Series.
        """
        if self._forecaster is None:
            raise ValueError("Forecaster has not been fitted yet. Call 'fit' first.")

        # Use stored forecasting horizon from Sktime.
        fh = self._fh if fh is None else fh

        forecast_df = self._forecaster.forecast.df_test
        # Use only the first len(fh) predictions.
        y_pred = pd.Series(
            forecast_df["forecast"].values[: len(fh)],
            index=forecast_df[self._forecaster.forecast.time_col][: len(fh)],
        )
        self._forecast = forecast_df
        return y_pred

    def get_fitted_params(self):
        """Return fitted parameters."""
        if self._forecaster is None:
            raise ValueError("Forecaster has not been fitted yet. Call 'fit' first.")
        return {
            "model": self._forecaster.model,
            "forecast_config": self.forecast_config,
        }
