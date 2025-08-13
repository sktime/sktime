# Copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Hurst Exponent Transformer for time series analysis."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from sktime.transformations.base import BaseTransformer


class HurstExponentTransformer(BaseTransformer):
    """Transformer for calculating the Hurst exponent of a time series.

    This transformer calculates the Hurst exponent, which is used to evaluate
    the auto-correlation properties of time series, particularly the degree of
    long-range dependence.

    Parameters
    ----------
    lags : Optional[Union[List[int], range]], default=None
        The lags to use for calculation. If None, uses a range based on min_lag
        and max_lag.
    method : str, default='rs'
        The method to use for Hurst exponent calculation. Either 'rs' (rescaled range)
        or 'dfa' (detrended fluctuation analysis).
    min_lag : int, default=2
        The minimum lag to use if lags is None.
    max_lag : int, default=100
        The maximum lag to use if lags is None.
    fit_trend : str, default='c'
        The trend component to include in the calculation.
    confidence_level : float, default=0.95
        The confidence level for the confidence interval calculation.

    Attributes
    ----------
    hurst_estimate_ : float
        The estimated Hurst exponent.
    confidence_interval_ : tuple
        The confidence interval for the Hurst exponent estimate.

    Examples
    --------
    >>> from sktime.transformations.series.hurst import HurstExponentTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = HurstExponentTransformer()
    >>> y_transform = transformer.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "None",
        "univariate-only": True,
        "requires_y": False,
        "fit_is_empty": False,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "capability:missing_values": False,
        "authors": ["phoeenniixx"],
    }

    def __init__(
        self,
        lags: Optional[Union[list[int], range]] = None,
        method: str = "rs",
        min_lag: int = 2,
        max_lag: int = 100,
        fit_trend: str = "c",
        confidence_level: float = 0.95,
    ):
        self.lags = lags
        self.method = method
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.fit_trend = fit_trend
        self.confidence_level = confidence_level

        super().__init__()

    def _fit(self, X: pd.Series, y=None):
        """Fit transformer to X and y.

        Parameters
        ----------
        X : pd.Series
            The time series data to fit.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        if len(X) < 4:  # Minimum length to calculate meaningful statistics
            raise ValueError(
                f"Time series too short. Length: {len(X)}, minimum required: 4"
            )

        self._effective_lags = self._get_effective_lags(len(X))

        try:
            self.hurst_estimate_ = self._hurst_exponent(X)
        except ValueError as e:
            raise ValueError(f"Failed to calculate Hurst exponent: {str(e)}") from e
        return self

    def _get_effective_lags(self, series_length: int) -> Union[list[int], range]:
        """Get effective lag range based on series length and initial parameters."""
        if self.lags is not None:
            return [lag for lag in self.lags if 2 <= lag <= series_length // 2]

        min_lag = max(2, self.min_lag)
        max_lag = min(series_length // 2, self.max_lag)

        if min_lag >= max_lag:
            min_lag = 2
            max_lag = max(4, series_length // 2)

        return range(min_lag, max_lag + 1)

    def _transform(self, X: pd.Series, y=None):
        """Transform X, return DataFrame with Hurst exponent and confidence interval.

        Parameters
        ----------
        X : pd.Series
            The time series data to transform.
        y : None
            Ignored.

        Returns
        -------
        X_transformed : pd.DataFrame
            A single-row DataFrame containing the Hurst exponent
            and confidence interval.
        """
        return pd.DataFrame(
            {
                "hurst_exponent": [self.hurst_estimate_],
                "confidence_interval_lower": [self.confidence_interval_[0]],
                "confidence_interval_upper": [self.confidence_interval_[1]],
            }
        )

    def _hurst_exponent(self, ts: pd.Series) -> float:
        """Calculate Hurst exponent for a single time series."""
        if self.method == "rs":
            return self._rs_method(ts)
        elif self.method == "dfa":
            return self._dfa_method(ts)
        else:
            raise ValueError("Invalid method. Choose 'rs' or 'dfa'.")

    def _rs_method(self, ts: pd.Series) -> float:
        """Rescaled range (R/S) method for Hurst exponent calculation."""
        tau = [self._calculate_rs(ts, lag) for lag in self._effective_lags]
        tau = [t for t in tau if np.isfinite(t)]  # Remove any NaN or infinite values
        if not tau:
            raise ValueError("Unable to calculate valid R/S values")
        return self._fit_hurst(self._effective_lags[: len(tau)], tau)

    def _dfa_method(self, ts: pd.Series) -> float:
        """Detrended Fluctuation Analysis method for Hurst exponent calculation."""
        tau = [self._calculate_dfa(ts, lag) for lag in self._effective_lags]
        tau = [t for t in tau if np.isfinite(t)]  # Remove any NaN or infinite values
        if not tau:
            raise ValueError("Unable to calculate valid DFA values")
        return self._fit_hurst(self._effective_lags[: len(tau)], tau)

    def _calculate_rs(self, ts: pd.Series, lag: int) -> float:
        """Calculate rescaled range for a given lag."""
        rolling_mean = ts.rolling(window=lag).mean()
        dev_from_mean = ts - rolling_mean
        range_ts = (
            dev_from_mean.rolling(window=lag).max()
            - dev_from_mean.rolling(window=lag).min()
        )
        std_ts = ts.rolling(window=lag).std()
        rs = range_ts / std_ts
        return np.nanmean(rs)

    def _calculate_dfa(self, ts: pd.Series, lag: int) -> float:
        """Calculate DFA fluctuation for a given lag."""
        y = ts.cumsum() - ts.mean()
        y_len = len(y)
        n_segments = y_len // lag
        fluctuations = []
        for i in range(n_segments):
            segment = y[i * lag : (i + 1) * lag]
            x = np.arange(lag)
            coef = np.polyfit(x, segment, 1)
            trend = np.polyval(coef, x)
            fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
            fluctuations.append(fluctuation)
        return np.mean(fluctuations)

    def _fit_hurst(self, lags: Union[list[int], range], tau: list[float]) -> float:
        """Fit Hurst exponent from log-log plot."""
        log_lags = np.log(lags)
        log_tau = np.log(tau)

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_lags, log_tau
        )

        self._calculate_confidence_interval(slope, std_err, len(lags))

        return slope

    def _calculate_confidence_interval(self, slope, std_err, n):
        """Calculate confidence interval for the Hurst exponent."""
        df = n - 2
        t_value = stats.t.ppf((1 + self.confidence_level) / 2, df)
        margin_of_error = t_value * std_err

        self.confidence_interval_ = (slope - margin_of_error, slope + margin_of_error)

    def plot_log_log(self, ts: pd.Series):
        """Plot the log-log graph used in Hurst exponent calculation."""
        from sktime.utils.dependencies._dependencies import _check_soft_dependencies

        try:
            _check_soft_dependencies("matplotlib", severity="warning")
            import matplotlib.pyplot as plt
        except ImportError:
            # If matplotlib is not installed, we'll just return without plotting
            return

        lags = self.lags or range(self.min_lag, min(self.max_lag, len(ts) // 2))
        if self.method == "rs":
            tau = [self._calculate_rs(ts, lag) for lag in lags]
            y_label = "log(R/S)"
        else:  # DFA method
            tau = [self._calculate_dfa(ts, lag) for lag in lags]
            y_label = "log(F)"

        log_lags = np.log(lags)
        log_tau = np.log(tau)

        plt.figure(figsize=(10, 6))
        plt.scatter(log_lags, log_tau, alpha=0.5)
        plt.plot(
            log_lags,
            self.hurst_estimate_ * log_lags
            + np.mean(log_tau - self.hurst_estimate_ * log_lags),
            color="r",
            label=f"Hurst = {self.hurst_estimate_:.3f}",
        )
        plt.xlabel("log(lag)")
        plt.ylabel(y_label)
        plt.title(
            f"Log-Log Plot for Hurst Exponent Estimation ({self.method.upper()} method)"
        )
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.show()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class
        """
        params1 = {"method": "rs", "min_lag": 2, "max_lag": 20}
        params2 = {"method": "dfa", "min_lag": 2, "max_lag": 20}
        return [params1, params2]
