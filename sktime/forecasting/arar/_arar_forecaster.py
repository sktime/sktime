# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""ARAR forecasting model."""

__all__ = ["ARARForecaster"]
__author__ = ["Akai01"]

import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from sktime.forecasting.base import BaseForecaster


def _setup_params(y_in, max_ar_depth=None, max_lag=None):
    """Set up default parameters based on series length.

    Parameters
    ----------
    y_in : array-like
        Input time series.
    max_ar_depth : int or None
        Maximum AR lag to consider.
    max_lag : int or None
        Maximum lag for autocovariances.

    Returns
    -------
    max_ar_depth : int
        Effective maximum AR depth.
    max_lag : int
        Effective maximum lag.
    """
    n = len(y_in)
    if n < 10:
        warnings.warn(
            f"Training data is too short (length={n}). The model may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    if max_ar_depth is None:
        if n > 40:
            max_ar_depth = 26
        elif n >= 13:
            max_ar_depth = 13
        else:  # 10 ≤ n < 13
            max_ar_depth = max(4, math.ceil(n / 3))

    if max_lag is None:
        if n > 40:
            max_lag = 40
        elif n >= 13:
            max_lag = 13
        else:
            max_lag = max(4, math.ceil(n / 2))
    return max_ar_depth, max_lag


def _fit_arar(y_in, max_ar_depth=None, max_lag=None, safe=True):
    """Fit ARAR model to a 1D series.

    Parameters
    ----------
    y_in : array-like
        Input time series.
    max_ar_depth : int or None
        Maximum AR lag to consider in subset selection.
    max_lag : int or None
        Maximum lag for computing autocovariances.
    safe : bool
        If True, return mean-based fallback on failure.

    Returns
    -------
    model_tuple : tuple
        (Y, best_phi, best_lag, sigma2, psi, sbar, max_ar_depth, max_lag)
        - Y: original series (np.ndarray, float)
        - best_phi: shape (4,) array for lags (1, i, j, k)
        - best_lag: tuple (1, i, j, k)
        - sigma2: innovation variance (float, >= 1e-12)
        - psi: memory-shortening filter (np.ndarray)
        - sbar: mean of shortened series (float)
        - max_ar_depth: effective max AR depth used
        - max_lag: effective max lag used
    """

    def mean_fallback(y):
        """Return a simple mean-based model as fallback."""
        mu = float(np.nanmean(y))
        var = float(np.nanvar(y, ddof=1)) if y.size > 1 else 0.0
        return (
            y.copy(),
            np.zeros(4, dtype=float),
            (1, 1, 1, 1),
            max(var, 1e-12),
            np.array([1.0], dtype=float),
            mu,
            max_ar_depth,
            max_lag,
        )

    try:
        y_in = np.asarray(y_in, dtype=float)
        Y = y_in.copy()

        # Setup parameters
        max_ar_depth, max_lag = _setup_params(
            y_in, max_ar_depth=max_ar_depth, max_lag=max_lag
        )

        # Quick guards
        if y_in.size < 5 or max_ar_depth < 4 or max_lag < max_ar_depth:
            if safe:
                return mean_fallback(y_in)
            raise ValueError("Too short series or incompatible max_ar_depth/max_lag")

        # --- Memory shortening (≤ 3 rounds) ---
        y = y_in.copy()
        psi = np.array([1.0], dtype=float)

        for _ in range(3):
            n = y.size
            taus = np.arange(1, min(15, n - 1) + 1, dtype=int)
            if taus.size == 0:
                break

            # One-step regressions for taus
            best_idx = None
            best_err = np.inf
            best_phi1 = 0.0
            for idx, t in enumerate(taus):
                den = float(np.dot(y[:-t], y[:-t])) + np.finfo(float).eps
                phi1 = float(np.dot(y[t:], y[:-t]) / den)
                den_e = float(np.dot(y[t:], y[t:])) + np.finfo(float).eps
                err = float(np.sum((y[t:] - phi1 * y[:-t]) ** 2) / den_e)
                if err < best_err:
                    best_err, best_idx, best_phi1 = err, idx, phi1

            tau = int(taus[best_idx])

            if best_err <= 8.0 / n or (best_phi1 >= 0.93 and tau > 2):
                # First-order reduction
                y = y[tau:] - best_phi1 * y[:-tau]
                psi = np.concatenate([psi, np.zeros(tau)]) - best_phi1 * np.concatenate(
                    [np.zeros(tau), psi]
                )
            elif best_phi1 >= 0.93:
                # Second-order reduction (use least squares)
                if n < 3:
                    break
                A = np.zeros((2, 2), dtype=float)
                A[0, 0] = float(np.dot(y[1 : n - 1], y[1 : n - 1]))
                A[0, 1] = A[1, 0] = float(np.dot(y[0 : n - 2], y[1 : n - 1]))
                A[1, 1] = float(np.dot(y[0 : n - 2], y[0 : n - 2]))
                b = np.array(
                    [
                        float(np.dot(y[2:n], y[1 : n - 1])),
                        float(np.dot(y[2:n], y[0 : n - 2])),
                    ],
                    dtype=float,
                )
                phi2, *_ = np.linalg.lstsq(A, b, rcond=None)
                y = y[2:n] - phi2[0] * y[1 : n - 1] - phi2[1] * y[0 : n - 2]
                psi = (
                    np.concatenate([psi, [0.0, 0.0]])
                    - phi2[0] * np.concatenate([[0.0], psi, [0.0]])
                    - phi2[1] * np.concatenate([[0.0, 0.0], psi])
                )
            else:
                break

        # Shortened series stats
        sbar = float(np.mean(y))
        X = y - sbar
        n = X.size

        # Compute gamma up to max_lag (biased)
        gamma = np.empty(max_lag + 1, dtype=float)
        xbar = float(np.mean(X))
        for lag in range(max_lag + 1):
            if lag >= n:
                gamma[lag] = 0.0
            else:
                gamma[lag] = float(np.sum((X[: n - lag] - xbar) * (X[lag:] - xbar)) / n)

        best_sigma2 = np.inf
        best_lag = (1, 0, 0, 0)
        best_phi = np.zeros(4, dtype=float)

        def build_system(i, j, k):
            """Build Yule-Walker system for given lags."""
            needed = [0, i - 1, j - 1, k - 1, j - i, k - i, k - j, 1, i, j, k]
            if any(idx < 0 or idx > max_lag for idx in needed):
                return None, None
            A = np.full((4, 4), gamma[0], dtype=float)  # diag = gamma[0]
            A[0, 1] = A[1, 0] = gamma[i - 1]
            A[0, 2] = A[2, 0] = gamma[j - 1]
            A[1, 2] = A[2, 1] = gamma[j - i]
            A[0, 3] = A[3, 0] = gamma[k - 1]
            A[1, 3] = A[3, 1] = gamma[k - i]
            A[2, 3] = A[3, 2] = gamma[k - j]
            b = np.array([gamma[1], gamma[i], gamma[j], gamma[k]], dtype=float)
            return A, b

        # Grid search over lag combinations
        for i in range(2, max_ar_depth - 1):
            for j in range(i + 1, max_ar_depth):
                for k in range(j + 1, max_ar_depth + 1):
                    A, b = build_system(i, j, k)
                    if A is None:
                        continue
                    phi, *_ = np.linalg.lstsq(A, b, rcond=None)
                    sigma2 = float(gamma[0] - float(np.dot(phi, b)))

                    # Accept any finite σ²
                    if np.isfinite(sigma2) and sigma2 < best_sigma2:
                        best_sigma2 = sigma2
                        best_phi = phi.astype(float, copy=True)
                        best_lag = (1, i, j, k)

        # Only fail if σ² is non-finite
        if not np.isfinite(best_sigma2):
            if safe:
                return mean_fallback(Y)
            raise RuntimeError("AR selection failed (no finite solution).")

        return (
            Y,
            best_phi.astype(float, copy=False),
            best_lag,
            max(best_sigma2, 1e-12),  # clamp for numerical safety
            psi.astype(float, copy=False),
            sbar,
            max_ar_depth,
            max_lag,
        )

    except Exception as e:
        if safe:
            return mean_fallback(np.asarray(y_in, dtype=float))
        raise RuntimeError(f"ARAR fitting failed: {e}") from e


def _forecast_arar(model_tuple, h, level=(80, 95)):
    """Forecast h steps ahead from an ARAR model.

    Parameters
    ----------
    model_tuple : tuple
        Model tuple from _fit_arar.
    h : int
        Forecast horizon (must be positive).
    level : tuple of float
        Confidence levels in percent, e.g. (80, 95).

    Returns
    -------
    forecast_dict : dict
        Dictionary with keys: mean, upper, lower, level

        - mean: (h,) forecasts
        - upper: (h, len(level)) upper bounds
        - lower: (h, len(level)) lower bounds
    """
    if h <= 0:
        raise ValueError("h must be positive")

    Y, best_phi, best_lag, sigma2, psi, sbar, _, _ = model_tuple
    Y = np.asarray(Y, dtype=float)
    best_phi = np.asarray(best_phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    sbar = float(sbar)
    sigma2 = float(sigma2)

    n = Y.size
    _, i, j, k = best_lag

    # Build xi (combined filter impulse response)
    def z(m):  # zeros helper
        return np.zeros(max(0, m), dtype=float)

    xi = np.concatenate([psi, z(k)])
    xi -= best_phi[0] * np.concatenate([[0.0], psi, z(k - 1)])
    xi -= best_phi[1] * np.concatenate([z(i), psi, z(k - i)])
    xi -= best_phi[2] * np.concatenate([z(j), psi, z(k - j)])
    xi -= best_phi[3] * np.concatenate([z(k), psi])

    # Iterative forecasts
    y_ext = np.concatenate([Y, np.zeros(h, dtype=float)])
    kk = xi.size
    c = (1.0 - float(np.sum(best_phi))) * sbar
    for t in range(1, h + 1):
        L = min(kk - 1, n + t - 1)
        y_ext[n + t - 1] = (
            (-np.dot(xi[1 : L + 1], y_ext[n + t - 1 - np.arange(1, L + 1)]) + c)
            if L > 0
            else c
        )
    mean_fc = y_ext[n : n + h].copy()

    # Extend xi for variance recursion
    if h > kk:
        xi = np.concatenate([xi, np.zeros(h - kk, dtype=float)])

    # Tau recursion for forecast error
    tau = np.zeros(h, dtype=float)
    tau[0] = 1.0
    for t in range(1, h):
        J = min(t, xi.size - 1)
        tau[t] = -np.dot(tau[:J], xi[1 : J + 1][::-1]) if J > 0 else 0.0

    se = np.sqrt(
        sigma2 * np.array([np.sum(tau[: t + 1] ** 2) for t in range(h)], dtype=float)
    )

    # Only compute bounds if levels are requested
    if len(level) > 0:
        zq = norm.ppf(0.5 + np.asarray(level) / 200.0)
        upper = np.column_stack([mean_fc + q * se for q in zq])
        lower = np.column_stack([mean_fc - q * se for q in zq])
    else:
        # Return empty arrays with correct shape when no levels requested
        upper = np.array([]).reshape(h, 0)
        lower = np.array([]).reshape(h, 0)

    return {"mean": mean_fc, "upper": upper, "lower": lower, "level": list(level)}


# ============================================================================
# ARARForecaster class
# ============================================================================


class ARARForecaster(BaseForecaster):
    r"""ARAR (AutoRegressive-AutoRegressive) forecaster.

    ARAR is a forecasting method that combines memory-shortening with
    subset autoregression. The method first applies a memory-shortening
    transformation to reduce long-term dependencies in the series, then
    fits a parsimonious autoregressive model using a subset of lags.

    The algorithm proceeds in two stages:

    1. Memory-shortening: Applies up to 3 rounds of filtering to reduce
       long-memory effects in the time series
    2. Subset AR: Selects an optimal subset of 3 lags from the shortened
       series using a grid search over possible lag combinations

    The ARAR model is a forecasting method designed for time series that may
    exhibit long-memory or persistent dependence. It works by automatically
    shortening the memory in the data and then fitting a small subset
    autoregressive (AR) model to the transformed series.

    Mathematical details follow about the two main stages of the ARAR algorithm:

    Stage 1: Memory Shortening (Adaptive AR Filter)

    The algorithm tests for long-memory structure by examining delayed
    correlations.

    * If long memory is detected, it applies a simple AR filter at the best delay.
    * This step may repeat up to three times, composing a filter

        :math:`\Psi(B) = 1 + \Psi_1 B + \cdots + \Psi_k B^k`

        until the transformed series behaves like a short-memory process.

    Stage 2: Subset AR Modeling

    After memory shortening, ARAR fits a **4-term subset AR model** using
    Yule-Walker equations. It searches over candidate lag sets and selects the
    model with the smallest estimated noise variance. The resulting AR polynomial

    :math:`\phi(B) = 1 - \phi_1 B - \phi_{l_1} B^{l_1} - \phi_{l_2} B^{l_2} - \phi_{l_3} B^{l_3}`  # noqa: E501

    combines with the memory-shortening filter to produce the full ARAR kernel

    :math:`\xi(B) = \Psi(B)\,\phi(B)`.

    This approach allows ARAR to automatically adapt to persistent dynamics while
    remaining computationally efficient. It often performs well on seasonal or
    slowly decaying series where pure ARMA or exponential-smoothing models struggle.

    Parameters
    ----------
    max_ar_depth : int or None, default=None
        Maximum AR lag to consider in subset selection.
        If None, defaults to:

        - 26 if n > 40
        - 13 if 13 <= n <= 40
        - max(4, ceil(n/3)) if n < 13

    max_lag : int or None, default=None
        Maximum lag for computing autocovariances.
        If None, defaults to:

        - 40 if n > 40
        - 13 if 13 <= n <= 40
        - max(4, ceil(n/2)) if n < 13

    safe : bool, default=True
        Whether to use safe fitting mode.
        * If True, returns a simple mean-based fallback model when fitting fails.
        * If False, raises an exception on failure.

    Attributes
    ----------
    model_ : tuple
        Fitted ARAR model containing:

        - Y: original series
        - best_phi: AR coefficients for selected lags
        - best_lag: tuple of selected AR lags (1, i, j, k)
        - sigma2: innovation variance
        - psi: memory-shortening filter
        - sbar: mean of shortened series
        - max_ar_depth: effective max AR depth used
        - max_lag: effective max lag used

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.arar import ARARForecaster
    >>> y = load_airline()
    >>> forecaster = ARARForecaster()
    >>> forecaster.fit(y)
    ARARForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])

    Prediction intervals and coefficients:
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.utils.plotting import plot_series
    >>>
    >>> # Load and split data
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=12)
    >>>
    >>> # Fit and predict
    >>> forecaster = ARARForecaster()
    >>> forecaster.fit(y_train)
    ARARForecaster(...)
    >>> y_pred = forecaster.predict(fh=list(range(1, 13)))
    >>> pred_int = forecaster.predict_interval(fh=list(range(1, 13)))
    >>>
    >>> # Plot results
    >>> plot_series(
    ...     y_train, y_test, y_pred, labels=["Train", "Test", "Forecast"],
    ...     title= "Forecast from Arar",
    ...     pred_int=pred_int
    ... )  # doctest: +SKIP
    >>>
    >>> # Print model information
    >>> print(f"Selected AR lags: {forecaster.model_[2]}")  # doctest: +SKIP
    >>> print(f"AR coefficients: {forecaster.model_[1]}")  # doctest: +SKIP
    >>> print(f"Innovation variance: {forecaster.model_[3]:.4f}")  # doctest: +SKIP

    References
    ----------
    .. [1] Brockwell, Peter J, and Richard A. Davis.
    Introduction to Time Series and Forecasting (2016), Chapter 10.
    """  # noqa: E501

    _tags = {
        # packaging info
        "authors": ["Akai01"],
        "maintainers": ["Akai01"],
        # estimator type
        "y_inner_mtype": "pd.Series",
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "capability:exogenous": False,
        "capability:missing_values": False,
        "capability:pred_int": True,
        "capability:pred_var": False,
        "capability:insample": False,
    }

    def __init__(self, max_ar_depth=None, max_lag=None, safe=True):
        self.max_ar_depth = max_ar_depth
        self.max_lag = max_lag
        self.safe = safe
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (ignored, not supported by ARAR).
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon (not required in fit).

        Returns
        -------
        self : reference to self
        """
        # Convert to numpy array for fitting
        y_np = y.values

        # Fit ARAR model
        self.model_ = _fit_arar(
            y_np, max_ar_depth=self.max_ar_depth, max_lag=self.max_lag, safe=self.safe
        )

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series (ignored, not supported by ARAR).

        Returns
        -------
        y_pred : pd.Series
            Point predictions for the forecast horizon.
        """
        # Get the forecast horizon as integer steps
        fh_int = fh.to_relative(self.cutoff)
        h = int(fh_int.max())

        # Generate forecasts
        forecast_dict = _forecast_arar(self.model_, h=h, level=())

        # Extract mean predictions for the requested horizon
        # Convert fh_int to numpy array for indexing (0-indexed)
        fh_idx = np.asarray(fh_int) - 1
        y_pred_values = forecast_dict["mean"][fh_idx]

        # Create index for predictions
        fh_abs = fh.to_absolute(self.cutoff)
        index = fh_abs.to_pandas()

        # Return as pandas Series
        return pd.Series(y_pred_values, index=index, name=self._y.name)

    def _predict_interval(self, fh, X, coverage):
        """Compute prediction intervals.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series (ignored, not supported by ARAR).
        coverage : float or list of float, optional (default=0.90)
            Nominal coverage(s) of the prediction intervals.

        Returns
        -------
        pred_int : pd.DataFrame
            Prediction intervals with columns for each coverage level.
        """
        # Convert to percentage
        level = [c * 100 for c in coverage]

        # Get the forecast horizon as integer steps
        fh_int = fh.to_relative(self.cutoff)
        h = int(fh_int.max())

        # Generate forecasts with intervals
        forecast_dict = _forecast_arar(self.model_, h=h, level=level)

        # Create index for predictions
        fh_abs = fh.to_absolute(self.cutoff)
        index = fh_abs.to_pandas()

        # Extract intervals for the requested horizon
        # Convert fh_int to numpy array for indexing (0-indexed)
        fh_idx = np.asarray(fh_int) - 1
        pred_int_dict = {}
        for i, cov in enumerate(coverage):
            lower_values = forecast_dict["lower"][fh_idx, i]
            upper_values = forecast_dict["upper"][fh_idx, i]

            pred_int_dict[(cov, "lower")] = pd.Series(
                lower_values, index=index, name=self._y.name
            )
            pred_int_dict[(cov, "upper")] = pd.Series(
                upper_values, index=index, name=self._y.name
            )

        # Create MultiIndex DataFrame
        cols = self._get_columns(method="predict_interval", coverage=coverage)
        index = fh.get_expected_pred_idx(y=self._y, cutoff=self.cutoff)
        pred_int = pd.DataFrame(pred_int_dict)
        pred_int.columns = cols
        pred_int.index = index

        return pred_int

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        params1 = {}
        params2 = {"max_ar_depth": 10, "max_lag": 15}
        return [params1, params2]
