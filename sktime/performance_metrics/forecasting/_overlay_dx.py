"""Overlay-dx: tolerance-sweep alignment score for point forecasting."""

__author__ = ["sktime developers"]
__all__ = ["OverlayDX"]

import numpy as np

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class OverlayDX(BaseForecastingErrorMetric):
    r"""Overlay-dx: tolerance-sweep alignment score for point forecasting.

    Output is a score in [0, 1] range (higher is better).

    Overlay-dx measures forecast alignment across multiple tolerance levels
    by computing the area under a tolerance-coverage curve, normalized by
    the maximum possible area. This provides a single, interpretable metric
    capturing "how well predictions align with actuals across tolerances."

    Mathematically, overlay_dx approximates the normalized integral of the
    empirical error coverage function, where coverage(τ) = percentage of
    predictions with absolute error ≤ τ.

    Parameters
    ----------
    tolerance_mode : {'range', 'quantile_range', 'absolute'}, default='range'
        How to define tolerance thresholds:

        * 'range': tolerance_τ = (τ% × (max(y_true) - min(y_true))) / 2
          Global range-based. The division by 2 ensures that a 100% tolerance
          corresponds to ±half the data range around each true value, mirroring
          common visual "band" or "ribbon" plot interpretations. This choice
          is heuristic but provides intuitive interpretation.

        * 'quantile_range': tolerance_τ = (τ% × (p95 - p05)) / 2
          Robust to outliers, uses interquantile range (5th to 95th percentile).
          Division by 2 maintains same interpretation as 'range' mode.

        * 'absolute': tolerance_τ = τ (raw units, no percentage conversion)
          For pre-normalized/scaled data where tolerance values are meaningful
          in absolute terms.

        NOTE: 'relative' mode (per-point tolerance) is not supported in v1
        due to algorithmic incompatibility. See GitHub issue for future work.

    max_tolerance_pct : float, default=100.0
        Maximum tolerance as percentage (or absolute if tolerance_mode='absolute').
        Must be > min_tolerance_pct.

    min_tolerance_pct : float, default=0.1
        Minimum tolerance as percentage (or absolute if tolerance_mode='absolute').
        Must be > 0.

    step_pct : float, default=0.1
        Step size for tolerance sweep (resolution parameter).

        Similar to bin width in histograms or number of bootstrap samples,
        this is a **discretization parameter**, not a modeling parameter.
        Smaller values → higher resolution but more computation.
        Default 0.1 gives ~1000 tolerance levels (high resolution).

        **IMPORTANT**: Scores are only comparable if computed with the same
        step_pct. When comparing forecasters, use identical parameters.

    multioutput : {'raw_values', 'uniform_average'} or array-like, default='uniform_average'
        For multivariate data:

        * 'uniform_average': average overlay_dx across all variables
        * 'raw_values': return overlay_dx per variable
        * array-like: weighted average with specified weights

        Note: Each variable has its own tolerance curve computed independently,
        then scores are aggregated.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}, \
            default='uniform_average'
        For hierarchical data - see BaseForecastingErrorMetric docs.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import OverlayDX
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> metric = OverlayDX()
    >>> metric(y_true, y_pred)  # doctest: +SKIP
    0.847

    Use with evaluate():

    >>> from sktime.performance_metrics.forecasting import evaluate
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.split import SlidingWindowSplitter
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster()
    >>> cv = SlidingWindowSplitter(fh=[1,2,3], window_length=48)
    >>> results = evaluate(forecaster, y, cv=cv, scoring=OverlayDX())  # doctest: +SKIP

    Use with grid search (will maximize score):

    >>> from sktime.forecasting.model_selection import ForecastingGridSearchCV
    >>> param_grid = {"sp": [1, 12]}
    >>> gscv = ForecastingGridSearchCV(  # doctest: +SKIP
    ...     forecaster=NaiveForecaster(),
    ...     cv=cv,
    ...     param_grid=param_grid,
    ...     scoring=OverlayDX()
    ... )

    Notes
    -----
    **Computational Complexity**: O(N log N + K) where N = number of points,
    K = number of tolerance steps ≈ (max_tol - min_tol) / step_pct.
    With default parameters, K ≈ 1000. For long time series or many CV folds,
    consider increasing step_pct to reduce computation.

    **Parameter Sensitivity**: The score depends on step_pct discretization.
    Only compare scores computed with identical parameters. The step_pct is
    included in the metric's __repr__ for reproducibility tracking.

    **Not Differentiable**: Uses indicator functions, so not suitable for
    gradient-based optimization. Works well with grid search or random search.

    **Constant Series**: If y_true is constant (range=0 or quantile_range=0):
    - Perfect match (y_pred = y_true everywhere) → score = 1.0
    - Any error → score = 0.0

    **Normalization vs Scale Invariance**:
    Output is normalized to [0, 1], but this does NOT imply scale invariance:
    - 'range': NOT scale-invariant (range changes with scale)
    - 'quantile_range': NOT scale-invariant (quantiles change with scale)
    - 'absolute': NOT scale-invariant (fixed tolerance in raw units)

    For scale-free comparison across series with different magnitudes,
    consider normalizing y_true and y_pred first.

    **by_index Not Supported**: overlay_dx is non-decomposable (coverage is
    a global property), making jackknife pseudo-values theoretically
    inappropriate and computationally expensive (O(N²)). The by_index
    parameter is not available for this metric.

    References
    ----------
    Based on Smile-SA/overlay_dx library.

    See Also
    --------
    MeanAbsolutePercentageError : Percentage-based error metric
    MeanAbsoluteScaledError : Scale-free error metric
    """

    _tags = {
        "object_type": ["metric_forecasting", "metric"],
        "scitype:y_pred": "pred",
        "requires-y-train": False,
        "requires-y-pred-benchmark": False,
        "capability:multivariate": True,
        "lower_is_better": False,  # This is a SCORE (higher is better)
        "inner_implements_multilevel": False,
        "reserved_params": ["multioutput", "multilevel"],
    }

    def __init__(
        self,
        tolerance_mode="range",
        max_tolerance_pct=100.0,
        min_tolerance_pct=0.1,
        step_pct=0.1,
        multioutput="uniform_average",
        multilevel="uniform_average",
    ):
        # Validate tolerance_mode
        allowed_modes = ["range", "quantile_range", "absolute"]
        if tolerance_mode not in allowed_modes:
            raise ValueError(
                f"tolerance_mode must be one of {allowed_modes}, "
                f"got {tolerance_mode!r}. "
                f"Note: 'relative' mode is not supported in v1 due to "
                f"algorithmic incompatibility (requires O(N×K) per-point "
                f"comparison). See GitHub issue for future work."
            )

        # Validate tolerance bounds
        if max_tolerance_pct <= min_tolerance_pct:
            raise ValueError(
                f"max_tolerance_pct ({max_tolerance_pct}) must be > "
                f"min_tolerance_pct ({min_tolerance_pct})"
            )

        if min_tolerance_pct <= 0:
            raise ValueError(
                f"min_tolerance_pct must be > 0, got {min_tolerance_pct}"
            )

        # Validate step size
        tolerance_range = max_tolerance_pct - min_tolerance_pct
        if step_pct <= 0 or step_pct > tolerance_range:
            raise ValueError(
                f"step_pct must be in (0, {tolerance_range}], got {step_pct}"
            )

        self.tolerance_mode = tolerance_mode
        self.max_tolerance_pct = max_tolerance_pct
        self.min_tolerance_pct = min_tolerance_pct
        self.step_pct = step_pct

        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
        )

        self.name = "OverlayDX"

    def __repr__(self):
        """Return string representation including key parameters."""
        return (
            f"OverlayDX("
            f"tolerance_mode={self.tolerance_mode!r}, "
            f"max_tolerance_pct={self.max_tolerance_pct}, "
            f"min_tolerance_pct={self.min_tolerance_pct}, "
            f"step_pct={self.step_pct})"
        )

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Core overlay_dx computation.

        Optimized implementation using sorted errors for O(N log N + K).

        Parameters
        ----------
        y_true : pandas DataFrame or Series
            Ground truth values.
        y_pred : pandas DataFrame or Series
            Predicted values.

        Returns
        -------
        float or ndarray
            Overlay-dx score(s).
        """
        multioutput = self.multioutput

        # Compute per-variable scores
        n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1

        if n_outputs == 1:
            # Univariate case
            y_t = y_true.values.flatten() if hasattr(y_true, "values") else y_true
            y_p = y_pred.values.flatten() if hasattr(y_pred, "values") else y_pred
            return self._compute_overlay_dx_single(y_t, y_p)
        else:
            # Multivariate case
            scores = np.zeros(n_outputs)
            for i in range(n_outputs):
                y_t = y_true.iloc[:, i].values
                y_p = y_pred.iloc[:, i].values
                scores[i] = self._compute_overlay_dx_single(y_t, y_p)

            # Handle multioutput aggregation
            if isinstance(multioutput, str) and multioutput == "raw_values":
                return scores
            elif isinstance(multioutput, str) and multioutput == "uniform_average":
                return np.mean(scores)
            else:
                # Weighted average
                return np.average(scores, weights=multioutput)

    def _compute_overlay_dx_single(self, y_true, y_pred):
        """Compute overlay_dx for a single univariate series.

        Optimized algorithm:
        1. Compute absolute errors
        2. Sort errors (O(N log N))
        3. Generate tolerance thresholds
        4. Use searchsorted for coverage (O(K log N))
        5. Integrate via trapezoidal rule
        6. Normalize by max area

        Parameters
        ----------
        y_true : array-like
            Ground truth values (1D).
        y_pred : array-like
            Predicted values (1D).

        Returns
        -------
        float
            Overlay-dx score in [0, 1].
        """
        # Compute absolute errors
        abs_errors = np.abs(y_true - y_pred)
        n = len(abs_errors)

        # Handle edge case: empty arrays
        if n == 0:
            raise ValueError(
                "Cannot compute OverlayDX score on empty arrays. "
                "y_true and y_pred must have at least one element."
            )

        # Handle edge case: NaN or Inf in data
        if np.any(np.isnan(abs_errors)) or np.any(np.isinf(abs_errors)):
            raise ValueError(
                "y_true and y_pred must not contain NaN or Inf values. "
                "Please clean your data before computing OverlayDX."
            )


        # Handle edge case: constant series
        if self.tolerance_mode in ["range", "quantile_range"]:
            if self.tolerance_mode == "range":
                value_range = np.max(y_true) - np.min(y_true)
            else:  # quantile_range
                # Check minimum sample size for meaningful quantiles
                if n < 20:
                    import warnings

                    warnings.warn(
                        f"Sample size ({n}) is small for quantile_range mode. "
                        f"Falling back to 'range' mode for this calculation. "
                        f"For reliable quantile estimates, use n >= 20.",
                        UserWarning,
                        stacklevel=3,
                    )
                    value_range = np.max(y_true) - np.min(y_true)
                else:
                    value_range = np.percentile(y_true, 95) - np.percentile(
                        y_true, 5
                    )

            # Constant series handling
            if value_range < np.finfo(np.float64).eps:
                # If range is zero, check if prediction matches
                if np.allclose(abs_errors, 0, atol=np.finfo(np.float64).eps):
                    return 1.0  # Perfect match
                else:
                    return 0.0  # Any error on constant series
        else:
            # For absolute mode, value_range is not used
            value_range = None

        # Sort errors once (O(N log N))
        sorted_errors = np.sort(abs_errors)

        # Generate tolerance thresholds
        tolerances = np.arange(
            self.min_tolerance_pct,
            self.max_tolerance_pct + self.step_pct,
            self.step_pct,
        )

        # Convert percentage tolerances to absolute tolerances
        abs_tolerances = self._pct_to_absolute_tolerance(tolerances, y_true, value_range)

        # Compute coverage at each tolerance (O(K log N))
        # coverage[i] = percentage of errors <= abs_tolerances[i]
        indices = np.searchsorted(sorted_errors, abs_tolerances, side="right")
        coverage_pct = (indices / n) * 100.0

        # Compute AUC using trapezoidal integration
        # X-axis: tolerance percentage (min_tol to max_tol)
        # Y-axis: coverage percentage (0 to 100)
        auc = np.trapz(coverage_pct, tolerances)

        # Normalize by maximum possible area
        # Max area = (max_tol - min_tol) × 100
        max_area = (self.max_tolerance_pct - self.min_tolerance_pct) * 100.0

        # Normalized score in [0, 1]
        score = auc / max_area

        return score

    def _pct_to_absolute_tolerance(self, pct_tolerances, y_true, value_range):
        """Convert percentage tolerances to absolute tolerances.

        Parameters
        ----------
        pct_tolerances : array-like
            Tolerance percentages.
        y_true : array-like
            Ground truth values (for reference in relative mode).
        value_range : float or None
            Value range (for range-based modes).

        Returns
        -------
        ndarray
            Absolute tolerance values.
        """
        if self.tolerance_mode == "range":
            # tolerance = (pct × range) / 2
            # The division by 2 ensures 100% tolerance = ±50% of range
            return (pct_tolerances / 100.0) * value_range / 2.0

        elif self.tolerance_mode == "quantile_range":
            # tolerance = (pct × quantile_range) / 2
            return (pct_tolerances / 100.0) * value_range / 2.0

        elif self.tolerance_mode == "absolute":
            # tolerance = pct (interpreted as absolute value, not percentage)
            return pct_tolerances

        else:
            raise ValueError(f"Unknown tolerance_mode: {self.tolerance_mode}")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances.
        """
        params1 = {}
        params2 = {"tolerance_mode": "quantile_range", "step_pct": 1.0}
        params3 = {"tolerance_mode": "absolute", "max_tolerance_pct": 10.0}

        return [params1, params2, params3]
