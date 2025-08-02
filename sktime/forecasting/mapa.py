"""MAPA Forecaster implementation."""

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies._dependencies import _check_soft_dependencies
from sktime.utils.warnings import warn


class MAPAForecaster(BaseForecaster):
    """MAPAForecaster implements the Multiple Aggregation Prediction Algorithm (MAPA).

    The MAPA method combines forecasts from different temporal aggregations of the time
    series to improve accuracy and robustness. It allows for multiple base
    forecasting methods and also supports various aggregation and combination
    strategies.

    Implementation Details:
    ----------------------
    The algorithm works in the following steps:

    1. Data Preparation:

        - Handles missing values using the specified imputation method
        - For multiplicative decomposition, ensures all values are positive by
          adding an offset if necessary

    2. For each aggregation level:

        - Aggregates the time series using the specified method (mean/sum)
        - Determines if seasonal decomposition should be enabled:

           - Calculates seasonal_period as sp // level
           - seasonal_enabled is True if all these conditions are met:

             * The original seasonal period (sp) is divisible by the level
             * The calculated seasonal_period is > 1
             * The time series length is >= 2 * seasonal_period

           - seasonal_enabled is False if level >= sp
        - Decomposes the series using STL decomposition:

           - Extracts trend using rolling averages if seasonal_enabled=False
           - Uses STLTransformer if seasonal_enabled=True
           - Stores seasonal patterns for later use

        - Fits the base forecaster on the trend component

    3. For prediction:

        - Generates forecasts using each level's base forecaster
        - If seasonal_enabled for that level:

           - Retrieves stored seasonal pattern
           - Applies seasonal adjustments to forecasts

        - Combines forecasts from all levels using specified method:

           - Simple mean
           - Median
           - Weighted mean (if weights provided)

        - Reverses any transformations applied during data preparation

    Based on R package: https://github.com/trnnick/mapa

    Parameters
    ----------
    aggregation_levels : list of int, default=None
        The levels at which the time series will be aggregated.
        If None, the levels will default to [1, 2, 4].

        For example, with daily data:

        - Level 1: Original daily data
        - Level 2: Aggregate every 2 days
        - Level 4: Aggregate every 4 days

        Lower levels capture short-term patterns while higher levels capture trends.

    base_forecaster : sktime-compatible forecaster, default=None
        The forecasting model to be used for each aggregation level.

        If None, defaults to:

        - ExponentialSmoothing(trend="add", seasonal="add", sp=sp) if statsmodel present
        - NaiveForecaster(strategy="mean") if statsmodel not present

    agg_method : str, default="mean"
        Method used to aggregate the time series at different temporal levels.

        Options are:

        - "mean": Takes average of the periods (e.g., average of each 2-day period)
        - "sum": Sums the values (useful for additive measures like sales)

    decompose_type : str, default="multiplicative"
        The type of decomposition used in time series decomposition.

        Options are:

        - "additive": Components are added (trend + seasonal + residual)
        - "multiplicative": Components are multiplied (trend * seasonal * residual)

    forecast_combine : str, default="mean"
        Method used to combine the forecasts from different aggregation levels.

        Options are:

        - "mean": Simple average of all forecasts
        - "median": Takes the median forecast
        - "weighted_mean": Uses supplied weights for weighted average

    imputation_method : str, default="ffill"
        Method used for imputing missing values in the time series.

        Options include:

        - "ffill": Forward fill (propagate last valid observation forward)
        - "bfill": Backward fill (use next valid observation)
        - "interpolate": Linear interpolation between valid observations

    sp : int, default=6
        Seasonal periodicity of the time series.

    weights : list of float, default=None
        Optional weights to apply when combining forecasts.
        Only used if forecast_combine="weighted_mean".
        Must have same length as aggregation_levels.
        Weights are normalized to sum to 1.
    """

    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": True,
        "authors": ["trnnick", "phoeenniixx", "satvshr"],
        "python_dependencies": ["statsmodels", "pandas>1"],
    }

    def __init__(
        self,
        aggregation_levels=None,
        base_forecaster=None,
        agg_method="mean",
        decompose_type="multiplicative",
        forecast_combine="mean",
        imputation_method="ffill",
        sp=6,
        weights=None,
    ):
        self.aggregation_levels = aggregation_levels
        self.agg_method = agg_method
        self.decompose_type = decompose_type
        self.forecast_combine = forecast_combine
        self.imputation_method = imputation_method
        self.sp = sp
        self.weights = weights
        self.base_forecaster = base_forecaster

        self._aggregation_levels = (
            self.aggregation_levels if self.aggregation_levels else [1, 2, 4]
        )

        self.forecasters = {}
        self._decomposition_info = {}
        self._y_cols = None
        self._y_name = None
        self._transformation_offset = None

        super().__init__()

        self._base_forecaster = self._initialize_base_forecaster(self.base_forecaster)

        if not all(
            isinstance(level, int) and level > 0 for level in self._aggregation_levels
        ):
            raise ValueError("All aggregation levels must be positive integers")

    def _initialize_base_forecaster(self, base_forecaster):
        """Initialize the base forecaster with appropriate fallbacks."""
        if base_forecaster is not None:
            return base_forecaster.clone()

        try:
            if _check_soft_dependencies("statsmodels", severity="none"):
                from sktime.forecasting.exp_smoothing import ExponentialSmoothing

                return ExponentialSmoothing(
                    trend="add",
                    seasonal="add",
                    sp=self.sp,
                    initialization_method="estimated",
                )
        except Exception as e:
            warn(
                f"Failed to initialize ExponentialSmoothing: {str(e)}. "
                "Falling back to NaiveForecaster."
            )

        from sktime.forecasting.naive import NaiveForecaster

        warn(
            "Using NaiveForecaster as base_forecaster. Install statsmodels for "
            "ExponentialSmoothing capability."
        )
        return NaiveForecaster(strategy="mean")

    def _handle_missing_data(self, y):
        if self.imputation_method == "ffill":
            y = y.ffill()
        elif self.imputation_method == "bfill":
            y = y.bfill()
        elif self.imputation_method == "interpolate":
            y = y.interpolate()
        else:
            raise ValueError(f"Unsupported imputation method: {self.imputation_method}")
        return y

    def _ensure_positive_values(self, y):
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        y = self._handle_missing_data(y)

        if self.decompose_type == "multiplicative" and (y <= 0).any().any():
            min_positive_value = y[y > 0].min().min()

            if pd.isna(min_positive_value) or min_positive_value <= 0:
                raise ValueError(
                    "Series must contain some strictly positive values\
                     for multiplicative decomposition."
                )

            offset = min_positive_value / 2
            y += offset
            self._transformation_offset = offset

            warn(
                f"Applied an offset of {offset} to ensure strictly positive values\
                             for multiplicative decomposition."
            )

        return y

    def _aggregate(self, y, level):
        """Aggregate the time series to the specified temporal level.

        Parameters
        ----------
        y : pandas.Series
            The input time series to be aggregated.

        level : int
            The aggregation level (e.g., 2 for bi-weekly, 4 for monthly).

        Returns
        -------
        pandas.Series
            The aggregated time series.
        """
        if level == 1:
            return y.copy()

        n_periods = len(y)
        groups = np.arange(n_periods) // level

        remainder = n_periods % level
        if remainder > 0:
            groups = groups[:-remainder]
            y_trimmed = y.iloc[:-remainder]
        else:
            y_trimmed = y

        if self.agg_method == "mean":
            aggregated = y_trimmed.groupby(groups).mean()
        elif self.agg_method == "sum":
            aggregated = y_trimmed.groupby(groups).sum()
        else:
            aggregated = y_trimmed.groupby(groups).agg(self.agg_method)

        if isinstance(y.index, pd.PeriodIndex):
            base_freq = y.index.freq.name
            if base_freq == "M":
                new_freq = f"{level}M"
            else:
                new_freq = f"{level}{base_freq}"
            try:
                new_index = pd.period_range(
                    start=y.index[0], periods=len(aggregated), freq=new_freq
                )
            except ValueError:
                new_index = pd.RangeIndex(start=0, stop=len(aggregated))
        else:
            new_index = pd.RangeIndex(start=0, stop=len(aggregated))

        aggregated.index = new_index
        return aggregated

    def _decompose(self, y, level):
        """Decompose the time series into trend, seasonal, and residual components.

        It uses STLTransformer.

        Parameters
        ----------
        y : pandas.DataFrame
            The input time series to be decomposed.
        level : int
            The aggregation level being processed.

        Returns
        -------
        tuple(decomposed_data, seasonal_enabled, seasonal_period)

            - decomposed_data: DataFrame containing trend, seasonal, residual components
            - seasonal_enabled: bool indicating if seasonal decomposition was performed
            - seasonal_period: int representing the seasonal period used
        """
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        seasonal_period = self.sp // level

        seasonal_enabled = (
            (self.sp % level == 0)
            and (seasonal_period > 1)
            and (len(y) >= 2 * seasonal_period)
        )

        if level >= self.sp:
            seasonal_enabled = False
            seasonal_period = 1

        self._decomposition_info[level] = {
            "seasonal_enabled": seasonal_enabled,
            "seasonal_period": seasonal_period,
            "n_observations": len(y),
        }

        decomposed = pd.DataFrame(index=y.index)

        for col in y.columns:
            series = y[col].copy()

            if not seasonal_enabled:
                trend = series.rolling(
                    window=min(len(series), seasonal_period * 2 + 1),
                    center=True,
                    min_periods=1,
                ).mean()

                trend = trend.ffill().bfill()

                if self.decompose_type == "multiplicative":
                    seasonal = pd.Series(1, index=series.index)
                    residual = series / trend
                else:  # additive
                    seasonal = pd.Series(0, index=series.index)
                    residual = series - trend

            else:
                if _check_soft_dependencies("statsmodels", severity="none"):
                    from sktime.transformations.series.detrend import STLTransformer
                stl = STLTransformer(
                    sp=seasonal_period,
                    seasonal=7,
                    trend=None,
                    low_pass=None,
                    seasonal_deg=1,
                    trend_deg=1,
                    low_pass_deg=1,
                    robust=False,
                    seasonal_jump=1,
                    trend_jump=1,
                    low_pass_jump=1,
                )

                stl.fit(series)

                trend = stl.trend_
                seasonal = stl.seasonal_
                residual = stl.resid_

                trend = trend.ffill().bfill()
                seasonal = seasonal.ffill().bfill()
                residual = residual.ffill().bfill()

                seasonal_pattern = pd.Series(
                    seasonal.values[:seasonal_period], index=range(seasonal_period)
                )
                if self.decompose_type == "multiplicative":
                    seasonal = np.exp(seasonal)
                    trend = np.exp(trend)
                    residual = series / (trend * seasonal)
                    seasonal_pattern = np.exp(seasonal_pattern)

                self._decomposition_info[level][f"{col}_seasonal_pattern"] = (
                    seasonal_pattern
                )

            decomposed[f"{col}_trend"] = trend
            decomposed[f"{col}_seasonal"] = seasonal
            decomposed[f"{col}_residual"] = residual

        return decomposed, seasonal_enabled, seasonal_period

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster following the MAPA methodology.

        It performs aggregation, decomposition, and fitting base forecasters at
        multiple levels.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Target time series to fit the forecaster.
        X : pd.DataFrame or None, optional (default=None)
            Exogenous variables for the forecasting model.
        fh : ForecastingHorizon or None, optional (default=None)
            Forecasting horizon for predictions

        Returns
        -------
        self : MAPAForecaster
            The fitted forecaster instance.
        """
        self._y_cols = (
            y.columns
            if isinstance(y, pd.DataFrame)
            else pd.Index([y.name])
            if y.name
            else pd.Index(["c0"])
        )
        self._y_name = y.name if isinstance(y, pd.Series) else None

        y = self._ensure_positive_values(y)

        valid_levels = []
        for level in self._aggregation_levels:
            try:
                y_agg = self._aggregate(y, level)
                y_agg.columns = self._y_cols

                decomposed, seasonal_enabled, seasonal_period = self._decompose(
                    y_agg, level
                )

                forecaster = type(self._base_forecaster)(
                    **self._base_forecaster.get_params()
                )

                if not seasonal_enabled:
                    forecaster_params = forecaster.get_params()
                    if "seasonal" in forecaster_params:
                        forecaster.set_params(seasonal=None)
                else:
                    forecaster_params = forecaster.get_params()
                    if "seasonal" in forecaster_params:
                        forecaster.set_params(seasonal="add", sp=seasonal_period)

                trend_cols = [
                    col for col in decomposed.columns if col.endswith("_trend")
                ]
                trend_data = decomposed[trend_cols].copy()
                trend_data.columns = self._y_cols

                forecaster.fit(trend_data, X=X, fh=fh)
                self.forecasters[level] = forecaster
                valid_levels.append(level)

            except Exception as e:
                warn(f"Failed to process level {level}: {str(e)}")
                continue

        if not valid_levels:
            raise ValueError("Failed to fit any aggregation levels")

        self._aggregation_levels = valid_levels
        return self

    def _predict(self, fh, X=None):
        """Generate forecasts based on the fitted MAPA forecaster.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon defining the steps ahead to predict.
        X : pd.DataFrame or None, optional (default=None)
            Exogenous variables for prediction.

        Returns
        -------
        pd.DataFrame
            Forecasted values for the specified forecasting horizon.
        """
        forecasts = []

        for level in self._aggregation_levels:
            try:
                info = self._decomposition_info.get(level, {})
                seasonal_enabled = info.get("seasonal_enabled", False)
                seasonal_period = info.get("seasonal_period", 1)

                if level not in self.forecasters:
                    warn(f"No forecaster found for level {level}")
                    continue

                forecast = self.forecasters[level].predict(fh, X)

                if isinstance(forecast, pd.Series):
                    forecast = pd.DataFrame(forecast)

                forecast.columns = self._y_cols

                if seasonal_enabled:
                    if isinstance(forecast.index, pd.PeriodIndex):
                        seasonal_idx = forecast.index.month % seasonal_period
                    else:
                        seasonal_idx = np.arange(len(forecast)) % seasonal_period

                    seasonal_factors = self._get_seasonal_pattern(level)

                    seasonal_adjustments = np.take(
                        seasonal_factors, seasonal_idx, mode="wrap"
                    )

                    if self.decompose_type == "multiplicative":
                        forecast = forecast.multiply(seasonal_adjustments, axis=0)
                    else:  # additive
                        forecast = forecast.add(seasonal_adjustments, axis=0)

                forecast_values = forecast.values
                if forecast_values.ndim == 1:
                    forecast_values = forecast_values.reshape(-1, 1)

                forecasts.append(forecast_values.ravel())

            except Exception as e:
                warn(f"Failed to generate forecast for level {level}: {str(e)}\n")
                continue

        if not forecasts:
            raise ValueError(
                "Failed to generate any forecasts. Check the following:\n"
                f"1. Valid levels: {self._aggregation_levels}\n"
                f"2. Decomposition info: {self._decomposition_info}\n"
                f"3. Available forecasters: {list(self.forecasters.keys())}"
            )

        forecasts = np.vstack(forecasts)
        final_forecast = self._combine_forecasts(forecasts)

        result = pd.DataFrame(
            final_forecast.reshape(-1, len(self._y_cols)),
            index=fh.to_absolute(self.cutoff).to_pandas(),
            columns=self._y_cols,
        )

        if hasattr(self, "_transformation_offset") and self._transformation_offset:
            if self.decompose_type == "multiplicative":
                result = result * (1 - self._transformation_offset)
            else:
                result = result - self._transformation_offset

        return result

    def _get_seasonal_pattern(self, level):
        """Extract seasonal pattern for a given aggregation level.

        Parameters
        ----------
        level : int
            Aggregation level

        Returns
        -------
        np.ndarray
            Seasonal pattern for the given level
        """
        info = self._decomposition_info.get(level, {})
        seasonal_period = info.get("seasonal_period", 1)

        if not info.get("seasonal_enabled", False):
            return (
                np.ones(seasonal_period)
                if self.decompose_type == "multiplicative"
                else np.zeros(seasonal_period)
            )

        forecaster = self.forecasters.get(level)
        if forecaster is None:
            return (
                np.ones(seasonal_period)
                if self.decompose_type == "multiplicative"
                else np.zeros(seasonal_period)
            )

        if hasattr(forecaster, "seasonal_"):
            pattern = forecaster.seasonal_
            if len(pattern) < seasonal_period:
                pattern = np.pad(
                    pattern, (0, seasonal_period - len(pattern)), mode="wrap"
                )
            return pattern[:seasonal_period]

        return (
            np.ones(seasonal_period)
            if self.decompose_type == "multiplicative"
            else np.zeros(seasonal_period)
        )

    def _combine_forecasts(self, forecasts):
        """Combine forecasts from multiple aggregation levels.

        Parameters
        ----------
        forecasts : np.ndarray
            Forecasts from different aggregation levels.

        Returns
        -------
        np.ndarray
            Combined forecast values.
        """
        if not isinstance(forecasts, np.ndarray):
            forecasts = np.array(forecasts)

        if self.weights is not None:
            if len(self.weights) != forecasts.shape[0]:
                raise ValueError(
                    "Weights must have the same length "
                    "as the number of aggregation levels."
                )
            weights = np.array(self.weights).reshape(-1, 1)
        else:
            weights = np.ones((forecasts.shape[0], 1))

        weights = weights / weights.sum()

        if self.forecast_combine == "mean":
            return np.mean(forecasts, axis=0)
        elif self.forecast_combine == "median":
            return np.median(forecasts, axis=0)
        elif self.forecast_combine == "weighted_mean":
            return np.average(forecasts, axis=0, weights=weights.ravel())
        else:
            raise ValueError(
                f"Unsupported forecast combination method: {self.forecast_combine}"
            )

    def _update(self, y, X=None, update_params=True):
        """Update with new data following MAPA methodology."""
        if isinstance(y, pd.Series):
            y = pd.DataFrame(y)
        if y.columns.empty:
            y.columns = self._y_cols

        y = self._ensure_positive_values(y)

        for level in self._aggregation_levels:
            try:
                y_agg = self._aggregate(y, level)
                y_agg.columns = self._y_cols

                if update_params:
                    if hasattr(self.forecasters[level], "update"):
                        self.forecasters[level].update(y_agg, X=X, update_params=True)
                    else:
                        self.forecasters[level].fit(y_agg, X=X)

            except Exception as e:
                warn(f"Failed to update level {level}: {str(e)}")
                continue

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster

        params = [
            {
                "aggregation_levels": [1, 2, 3],
                "base_forecaster": NaiveForecaster(strategy="mean"),
                "imputation_method": "ffill",
                "decompose_type": "multiplicative",
                "forecast_combine": "mean",
            },
            {
                "aggregation_levels": [1, 3, 6],
                "base_forecaster": PolynomialTrendForecaster(degree=2),
                "imputation_method": "ffill",
                "decompose_type": "multiplicative",
                "forecast_combine": "weighted_mean",
                "weights": [0.5, 0.3, 0.2],
            },
        ]
        if _check_soft_dependencies("statsmodels", severity="none"):
            from sktime.forecasting.exp_smoothing import ExponentialSmoothing

            params.append(
                {
                    "aggregation_levels": [1, 4, 6],
                    "base_forecaster": ExponentialSmoothing(
                        trend="add", seasonal="mul", sp=6
                    ),
                    "imputation_method": "interpolate",
                    "decompose_type": "additive",
                    "forecast_combine": "median",
                }
            )

        return params
