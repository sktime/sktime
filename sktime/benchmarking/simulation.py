# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Time series data simulation for benchmarking.

This module provides utilities for generating synthetic time series data
with various statistical distributions and custom patterns.
"""

__author__ = ["IndarKumar"]
__all__ = ["TimeSeriesSimulator"]


import numpy as np
import pandas as pd

from sktime.base import BaseObject


class TimeSeriesSimulator(BaseObject):
    """Generate synthetic univariate time series with specified distributions.

    The TimeSeriesSimulator generates synthetic time series data that follows
    a specified statistical distribution or custom pattern. This is useful for
    benchmarking forecasting models against known data characteristics.

    The generated series can be used directly with sktime's ForecastingBenchmark
    to evaluate how different models perform on data with specific properties.

    Parameters
    ----------
    length : int, optional, default=100
        Length of the time series to generate. If seasonality is specified and
        length would be less than 3 times the maximum seasonal period, length
        will be automatically increased to ensure at least 3 full seasonal cycles.
    distribution : str or callable, default="normal"
        Distribution to use for generating the time series values.
        If str, must be one of:

        - "normal" : Normal/Gaussian distribution
        - "poisson" : Poisson distribution
        - "exponential" : Exponential distribution
        - "gamma" : Gamma distribution
        - "uniform" : Uniform distribution
        - "binomial" : Binomial distribution
        - "lognormal" : Log-normal distribution

        If callable, must accept parameters (size, random_state) and return array.
    dist_params : dict, optional, default=None
        Parameters to pass to the distribution function.
        For built-in distributions:

        - "normal": {"loc": 0, "scale": 1}
        - "poisson": {"lam": 5}
        - "exponential": {"scale": 1.0}
        - "gamma": {"shape": 2.0, "scale": 2.0}
        - "uniform": {"low": 0, "high": 1}
        - "binomial": {"n": 10, "p": 0.5}
        - "lognormal": {"mean": 0, "sigma": 1}

    trend : str or callable, optional, default=None
        Trend component to add to the series.
        If str, must be one of: "linear", "quadratic", "exponential".
        If callable, must accept array of time indices and return trend values.
    trend_params : dict, optional, default=None
        Parameters for the trend function.

        - For "linear": {"slope": 1.0, "intercept": 0.0}
        - For "quadratic": {"a": 0.01, "b": 0, "c": 0}
        - For "exponential": {"base": 1.01, "scale": 1.0}

    seasonality : int or list of int, optional, default=None
        Seasonal period(s) to add to the series. Can be a single integer
        or list of integers for multiple seasonal components.
    seasonality_strength : float or list of float, optional, default=1.0
        Amplitude of seasonal component(s). If list, must match length
        of seasonality parameter.
    noise_std : float, optional, default=0.0
        Standard deviation of Gaussian noise to add to the series.
    freq : str, optional, default="D"
        Frequency of the time index. Pandas frequency string.
    start : str or pd.Timestamp, optional, default=None
        Start time for the time index. If None, defaults to "2020-01-01".
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    time_index_ : pd.DatetimeIndex
        Generated time index for the series, set after calling simulate.

    See Also
    --------
    sktime.benchmarking.forecasting.ForecastingBenchmark :
        Benchmarking framework that can use simulated data.

    Examples
    --------
    >>> from sktime.benchmarking.simulation import TimeSeriesSimulator
    >>> # Generate Poisson distributed time series
    >>> sim = TimeSeriesSimulator(
    ...     length=100,
    ...     distribution="poisson",
    ...     dist_params={"lam": 10},
    ...     random_state=42,
    ... )
    >>> y = sim.simulate()

    >>> # Generate time series with custom distribution
    >>> def custom_dist(size, random_state):
    ...     return random_state.beta(2, 5, size=size)
    >>> sim = TimeSeriesSimulator(
    ...     length=100,
    ...     distribution=custom_dist,
    ...     random_state=42,
    ... )
    >>> y = sim.simulate()

    >>> # Generate time series with trend and seasonality
    >>> sim = TimeSeriesSimulator(
    ...     length=365,
    ...     distribution="normal",
    ...     dist_params={"loc": 50, "scale": 10},
    ...     trend="linear",
    ...     trend_params={"slope": 0.1},
    ...     seasonality=7,
    ...     seasonality_strength=5.0,
    ...     random_state=42,
    ... )
    >>> y = sim.simulate()

    >>> # Use with ForecastingBenchmark
    >>> from sktime.benchmarking.forecasting import ForecastingBenchmark
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.performance_metrics.forecasting import MeanAbsoluteError
    >>> from sktime.split import ExpandingWindowSplitter
    >>>
    >>> # Create simulated data
    >>> sim = TimeSeriesSimulator(
    ...     length=100,
    ...     distribution="poisson",
    ...     dist_params={"lam": 10},
    ...     random_state=42,
    ... )
    >>> y = sim.simulate()
    >>>
    >>> # Use simulated data in ForecastingBenchmark
    >>> benchmark = ForecastingBenchmark()
    >>> benchmark.add_estimator(NaiveForecaster())
    >>> benchmark.add_task(
    ...     dataset_loader=y,
    ...     cv_splitter=ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=50),
    ...     scorers=[MeanAbsoluteError()],
    ... )  # doctest: +SKIP
    """

    _tags = {
        "authors": ["IndarKumar"],
        "maintainers": ["IndarKumar"],
        "object_type": "simulator",
        "python_dependencies": None,
    }

    # Built-in distribution functions
    _DISTRIBUTIONS = {
        "normal",
        "poisson",
        "exponential",
        "gamma",
        "uniform",
        "binomial",
        "lognormal",
    }

    def __init__(
        self,
        length=100,
        distribution="normal",
        dist_params=None,
        trend=None,
        trend_params=None,
        seasonality=None,
        seasonality_strength=1.0,
        noise_std=0.0,
        freq="D",
        start=None,
        random_state=None,
    ):
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")

        if seasonality is not None:
            if isinstance(seasonality, (list, tuple)):
                max_seasonality = max(seasonality)
            else:
                max_seasonality = seasonality
            min_length = 3 * max_seasonality
            if length < min_length:
                length = min_length

        self.length = length
        self.distribution = distribution
        self.dist_params = dist_params
        self.trend = trend
        self.trend_params = trend_params
        self.seasonality = seasonality
        self.seasonality_strength = seasonality_strength
        self.noise_std = noise_std
        self.freq = freq
        self.start = start
        self.random_state = random_state

        super().__init__()

    def simulate(self):
        """Generate synthetic time series.

        Returns
        -------
        y : pd.Series
            Simulated time series with datetime index.

        Examples
        --------
        >>> from sktime.benchmarking.simulation import TimeSeriesSimulator
        >>> sim = TimeSeriesSimulator(
        ...     length=50,
        ...     distribution="normal",
        ...     random_state=42,
        ... )
        >>> y = sim.simulate()
        >>> len(y)
        50
        """
        from sklearn.utils import check_random_state

        self._random_state = check_random_state(self.random_state)

        start_time = self.start if self.start is not None else "2020-01-01"
        self.time_index_ = pd.date_range(
            start=start_time, periods=self.length, freq=self.freq
        )

        values = self._generate_from_distribution()

        if self.trend is not None:
            values = values + self._generate_trend()

        if self.seasonality is not None:
            values = values + self._generate_seasonality()

        if self.noise_std > 0:
            noise = self._random_state.normal(0, self.noise_std, size=self.length)
            values = values + noise

        y = pd.Series(values, index=self.time_index_, name="simulated")

        return y

    def _generate_from_distribution(self):
        """Generate values from the specified distribution.

        Returns
        -------
        values : np.ndarray
            Generated values from the distribution.
        """
        # Get distribution parameters
        params = self.dist_params if self.dist_params is not None else {}

        if callable(self.distribution):
            # Custom distribution function
            values = self.distribution(
                size=self.length, random_state=self._random_state
            )
        elif self.distribution == "normal":
            loc = params.get("loc", 0)
            scale = params.get("scale", 1)
            values = self._random_state.normal(loc, scale, size=self.length)
        elif self.distribution == "poisson":
            lam = params.get("lam", 5)
            values = self._random_state.poisson(lam, size=self.length).astype(float)
        elif self.distribution == "exponential":
            scale = params.get("scale", 1.0)
            values = self._random_state.exponential(scale, size=self.length)
        elif self.distribution == "gamma":
            shape = params.get("shape", 2.0)
            scale = params.get("scale", 2.0)
            values = self._random_state.gamma(shape, scale, size=self.length)
        elif self.distribution == "uniform":
            low = params.get("low", 0)
            high = params.get("high", 1)
            values = self._random_state.uniform(low, high, size=self.length)
        elif self.distribution == "binomial":
            n = params.get("n", 10)
            p = params.get("p", 0.5)
            values = self._random_state.binomial(n, p, size=self.length).astype(float)
        elif self.distribution == "lognormal":
            mean = params.get("mean", 0)
            sigma = params.get("sigma", 1)
            values = self._random_state.lognormal(mean, sigma, size=self.length)
        else:
            raise ValueError(
                f"Unknown distribution '{self.distribution}'. "
                f"Expected one of {sorted(self._DISTRIBUTIONS)} or a callable."
            )

        return values

    def _generate_trend(self):
        """Generate trend component.

        Returns
        -------
        trend : np.ndarray
            Trend values.
        """
        t = np.arange(self.length)
        params = self.trend_params if self.trend_params is not None else {}

        if callable(self.trend):
            # Custom trend function
            trend = self.trend(t)
        elif self.trend == "linear":
            slope = params.get("slope", 1.0)
            intercept = params.get("intercept", 0.0)
            trend = slope * t + intercept
        elif self.trend == "quadratic":
            a = params.get("a", 0.01)
            b = params.get("b", 0)
            c = params.get("c", 0)
            trend = a * t**2 + b * t + c
        elif self.trend == "exponential":
            base = params.get("base", 1.01)
            scale = params.get("scale", 1.0)
            trend = scale * (base**t - 1)
        else:
            raise ValueError(
                f"Unknown trend: {self.trend}. "
                "Must be 'linear', 'quadratic', 'exponential', or a callable."
            )

        return trend

    def _generate_seasonality(self):
        """Generate seasonal component(s).

        Returns
        -------
        seasonality : np.ndarray
            Seasonal values.
        """
        t = np.arange(self.length)
        seasonal_component = np.zeros(self.length)

        # Handle single or multiple seasonal periods
        periods = (
            [self.seasonality]
            if isinstance(self.seasonality, int)
            else self.seasonality
        )
        strengths = (
            [self.seasonality_strength]
            if isinstance(self.seasonality_strength, (int, float))
            else self.seasonality_strength
        )

        if len(strengths) == 1 and len(periods) > 1:
            # Broadcast single strength to all periods
            strengths = strengths * len(periods)
        elif len(strengths) != len(periods):
            raise ValueError(
                "seasonality_strength must be a single value or match "
                "the number of seasonal periods."
            )

        for period, strength in zip(periods, strengths):
            seasonal_component += strength * np.sin(2 * np.pi * t / period)

        return seasonal_component

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {
            "length": 50,
            "distribution": "normal",
            "dist_params": {"loc": 0, "scale": 1},
            "random_state": 42,
        }
        params2 = {
            "length": 100,
            "distribution": "poisson",
            "dist_params": {"lam": 10},
            "trend": "linear",
            "trend_params": {"slope": 0.1},
            "seasonality": 7,
            "seasonality_strength": 2.0,
            "random_state": 42,
        }
        return [params1, params2]
