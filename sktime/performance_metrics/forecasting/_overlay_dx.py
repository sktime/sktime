"""Overlay-dx: tolerance-sweep alignment score for point forecasting."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Utkarshkarki"]
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
    empirical error coverage function, where ``coverage(τ)`` = percentage of
    predictions with absolute error ≤ τ.

    Parameters
    ----------
    tolerance_mode : {'range', 'quantile_range', 'absolute', 'relative'}, \
            default='range'
        How to define tolerance thresholds:

        * ``'range'``: ``tolerance_τ = (τ% × (max(y_true) - min(y_true))) / 2``
          Global range-based. The division by 2 ensures that a 100% tolerance
          corresponds to ±half the data range around each true value, mirroring
          common visual "band" or "ribbon" plot interpretations.

        * ``'quantile_range'``: ``tolerance_τ = (τ% × (p95 - p05)) / 2``
          Robust to outliers; uses interquantile range (5th to 95th percentile).
          Division by 2 maintains the same interpretation as ``'range'`` mode.

        * ``'absolute'``: ``tolerance_τ = τ`` (raw units, no percentage conversion).
          For pre-normalized/scaled data where tolerance values are meaningful
          in absolute terms.

        * ``'relative'``: ``tolerance_{τ,i} = (τ% × |y_true_i|)``
          **Per-point** relative tolerance. Each data point has its own tolerance
          proportional to its true value. Useful for comparing forecasts across
          series with different scales, or when percentage-based accuracy
          matters (e.g., finance, growth metrics, economics).

          .. note::
             ``'relative'`` mode uses an O(N×K) pointwise comparison algorithm
             and cannot reuse the O(N log N + K) sorted-array approach of other
             modes. For large N and K this may be slower. See Notes section.

    max_tolerance_pct : float, default=100.0
        Maximum tolerance as percentage (or absolute value if
        ``tolerance_mode='absolute'``). Must be > ``min_tolerance_pct``.

    min_tolerance_pct : float, default=0.1
        Minimum tolerance as percentage (or absolute value if
        ``tolerance_mode='absolute'``). Must be > 0.

    step_pct : float, default=0.1
        Step size for tolerance sweep (resolution parameter).

        Similar to bin width in histograms, this is a **discretization
        parameter**, not a modeling parameter. Smaller values give higher
        resolution but more computation. Default 0.1 gives ~1000 tolerance
        levels (high resolution).

        .. important::
           Scores are only comparable if computed with the same ``step_pct``.
           When comparing forecasters, use identical parameters.

    multioutput : {'raw_values', 'uniform_average'} or array-like, \
            default='uniform_average'
        For multivariate data:

        * ``'uniform_average'``: average overlay_dx across all variables.
        * ``'raw_values'``: return overlay_dx per variable.
        * array-like: weighted average with specified weights.

        Each variable has its own tolerance curve computed independently,
        then scores are aggregated.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}, \
            default='uniform_average'
        For hierarchical data - see ``BaseForecastingErrorMetric`` docs.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import OverlayDX
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> metric = OverlayDX()
    >>> metric(y_true, y_pred)  # doctest: +SKIP
    0.847

    Using relative tolerance mode (per-point tolerances):

    >>> metric_rel = OverlayDX(tolerance_mode="relative")
    >>> metric_rel(y_true, y_pred)  # doctest: +SKIP
    0.712

    Notes
    -----
    **Computational Complexity**:

    * ``'range'``, ``'quantile_range'``, ``'absolute'``: O(N log N + K), where
      N = number of points, K = number of tolerance steps ≈
      (max_tol - min_tol) / step_pct. Errors are sorted once; coverage at each
      threshold is found via binary search (``np.searchsorted``).

    * ``'relative'``: O(N × K). Per-point tolerances break the sorted-array
      invariant, requiring a full pass over all N points for each of K
      thresholds. For large datasets or many tolerance steps, consider
      increasing ``step_pct`` to reduce K.

    **Parameter Sensitivity**: The score depends on ``step_pct``
    discretization. Only compare scores computed with identical parameters.

    **Not Differentiable**: Uses indicator functions, so not suitable for
    gradient-based optimization. Works well with grid search or random search.

    **Constant Series**: If ``y_true`` is constant (range = 0 or
    quantile_range = 0):

    * Perfect match (``y_pred = y_true`` everywhere) → score = 1.0
    * Any error → score = 0.0

    **Zero values in relative mode**: When ``|y_true_i| = 0``, the per-point
    tolerance is 0 at every threshold τ, so point i is covered only if
    ``abs_error_i = 0`` exactly. Series with many zero true values may yield
    low scores even for small errors.

    **Normalization vs Scale Invariance**:
    Output is normalized to [0, 1], but this does NOT imply scale invariance:

    * ``'range'``: NOT scale-invariant (range changes with scale).
    * ``'quantile_range'``: NOT scale-invariant (quantiles change with scale).
    * ``'absolute'``: NOT scale-invariant (fixed tolerance in raw units).
    * ``'relative'``: scale-invariant (tolerance scales with ``|y_true_i|``).

    **by_index Not Supported**: overlay_dx is non-decomposable (coverage is
    a global property).

    See Also
    --------
    MeanAbsolutePercentageError : Percentage-based error metric.
    MeanAbsoluteScaledError : Scale-free error metric.
    """

    _tags = {
        "object_type": ["metric_forecasting", "metric"],
        "scitype:y_pred": "pred",
        "requires-y-train": False,
        "requires-y-pred-benchmark": False,
        "capability:multivariate": True,
        "lower_is_better": False,  # SCORE: higher is better
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
        allowed_modes = ["range", "quantile_range", "absolute", "relative"]
        if tolerance_mode not in allowed_modes:
            raise ValueError(
                f"tolerance_mode must be one of {allowed_modes}, "
                f"got {tolerance_mode!r}."
            )

        if max_tolerance_pct <= min_tolerance_pct:
            raise ValueError(
                f"max_tolerance_pct ({max_tolerance_pct}) must be > "
                f"min_tolerance_pct ({min_tolerance_pct})"
            )

        if min_tolerance_pct <= 0:
            raise ValueError(
                f"min_tolerance_pct must be > 0, got {min_tolerance_pct}"
            )

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

        Dispatches to the appropriate algorithm based on ``tolerance_mode``:

        * Non-relative modes: O(N log N + K) via sorted errors +
          ``np.searchsorted``.
        * Relative mode: O(N × K) via per-point pointwise comparison.

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

        n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1

        if n_outputs == 1:
            y_t = y_true.values.flatten() if hasattr(y_true, "values") else y_true
            y_p = y_pred.values.flatten() if hasattr(y_pred, "values") else y_pred
            return self._compute_overlay_dx_single(y_t, y_p)
        else:
            scores = np.zeros(n_outputs)
            for i in range(n_outputs):
                y_t = y_true.iloc[:, i].values
                y_p = y_pred.iloc[:, i].values
                scores[i] = self._compute_overlay_dx_single(y_t, y_p)

            if isinstance(multioutput, str) and multioutput == "raw_values":
                return scores
            elif isinstance(multioutput, str) and multioutput == "uniform_average":
                return np.mean(scores)
            else:
                return np.average(scores, weights=multioutput)

    def _compute_overlay_dx_single(self, y_true, y_pred):
        """Compute overlay_dx for a single univariate series.

        Dispatches to the optimized O(N log N + K) algorithm for global
        tolerance modes, or the O(N × K) algorithm for relative mode.

        Parameters
        ----------
        y_true : array-like, shape (N,)
            Ground truth values (1D).
        y_pred : array-like, shape (N,)
            Predicted values (1D).

        Returns
        -------
        float
            Overlay-dx score in [0, 1].
        """
        abs_errors = np.abs(y_true - y_pred)
        n = len(abs_errors)

        if n == 0:
            raise ValueError(
                "Cannot compute OverlayDX score on empty arrays. "
                "y_true and y_pred must have at least one element."
            )

        if np.any(np.isnan(abs_errors)) or np.any(np.isinf(abs_errors)):
            raise ValueError(
                "y_true and y_pred must not contain NaN or Inf values. "
                "Please clean your data before computing OverlayDX."
            )

        if self.tolerance_mode == "relative":
            return self._compute_overlay_dx_relative(y_true, abs_errors)

        # --- Global-tolerance modes: O(N log N + K) ---
        value_range = self._compute_value_range(y_true, n)

        if value_range is not None and value_range < np.finfo(np.float64).eps:
            # Constant series
            if np.allclose(abs_errors, 0, atol=np.finfo(np.float64).eps):
                return 1.0
            else:
                return 0.0

        # Sort errors once – O(N log N)
        sorted_errors = np.sort(abs_errors)

        tolerances = np.arange(
            self.min_tolerance_pct,
            self.max_tolerance_pct + self.step_pct,
            self.step_pct,
        )

        abs_tolerances = self._pct_to_absolute_tolerance(
            tolerances, y_true, value_range
        )

        # Coverage at each threshold via binary search – O(K log N)
        indices = np.searchsorted(sorted_errors, abs_tolerances, side="right")
        coverage_pct = (indices / n) * 100.0

        auc = _trapezoid(coverage_pct, tolerances)
        max_area = (self.max_tolerance_pct - self.min_tolerance_pct) * 100.0
        # Clamp to [0, 1]: trapezoid rule can overshoot by a tiny epsilon
        # when coverage is 100% at every grid point (e.g. perfect prediction).
        return float(np.clip(auc / max_area, 0.0, 1.0))

    def _compute_overlay_dx_relative(self, y_true, abs_errors):
        """Compute overlay_dx using per-point relative tolerances – O(N × K).

        For each tolerance threshold τ, the per-point tolerance is:

        .. math::

            \\text{tolerance}_{\\tau,i} = \\frac{\\tau}{100} \\times |y_{\\text{true},i}|

        A prediction at index i is considered *covered* at threshold τ if:

        .. math::

            |y_{\\text{true},i} - y_{\\text{pred},i}|
            \\leq \\text{tolerance}_{\\tau,i}

        Because tolerances differ per point, the sorted-array invariant of the
        global modes is broken; we must check all N points for every threshold.

        **Complexity**: O(N × K), where K = len(tolerances).

        Parameters
        ----------
        y_true : ndarray, shape (N,)
            Ground truth values (1D).
        abs_errors : ndarray, shape (N,)
            Pre-computed absolute errors ``|y_true - y_pred|``.

        Returns
        -------
        float
            Overlay-dx score in [0, 1].

        Notes
        -----
        When ``|y_true_i| = 0``, the per-point tolerance is 0 at all τ, so
        point i contributes to coverage only when its absolute error is
        exactly 0.
        """
        abs_y_true = np.abs(y_true)  # (N,)

        tolerances = np.arange(
            self.min_tolerance_pct,
            self.max_tolerance_pct + self.step_pct,
            self.step_pct,
        )

        coverage_pct = np.empty(len(tolerances))

        # Vectorised inner loop: broadcast (K,) tolerances against (N,) arrays
        # shape: (K, 1) * (1, N) → (K, N) – check all N points at all K thresholds
        tol_fractions = (tolerances / 100.0)[:, np.newaxis]  # (K, 1)
        tol_matrix = tol_fractions * abs_y_true[np.newaxis, :]  # (K, N)
        covered_matrix = abs_errors[np.newaxis, :] <= tol_matrix  # (K, N) bool
        coverage_pct = covered_matrix.mean(axis=1) * 100.0  # (K,)

        auc = _trapezoid(coverage_pct, tolerances)
        max_area = (self.max_tolerance_pct - self.min_tolerance_pct) * 100.0
        # Clamp to [0, 1]: trapezoid rule can overshoot by a tiny epsilon
        # when coverage is 100% at every grid point (e.g. perfect prediction).
        return float(np.clip(auc / max_area, 0.0, 1.0))

    def _compute_value_range(self, y_true, n):
        """Compute the value range used to scale percentage tolerances.

        Parameters
        ----------
        y_true : ndarray, shape (N,)
            Ground truth values.
        n : int
            Length of y_true.

        Returns
        -------
        float or None
            Value range for range/quantile_range modes; ``None`` for absolute.
        """
        if self.tolerance_mode == "range":
            return np.max(y_true) - np.min(y_true)

        elif self.tolerance_mode == "quantile_range":
            if n < 20:
                import warnings

                warnings.warn(
                    f"Sample size ({n}) is small for quantile_range mode. "
                    f"Falling back to 'range' mode for this calculation. "
                    f"For reliable quantile estimates, use n >= 20.",
                    UserWarning,
                    stacklevel=5,
                )
                return np.max(y_true) - np.min(y_true)
            return np.percentile(y_true, 95) - np.percentile(y_true, 5)

        elif self.tolerance_mode == "absolute":
            return None  # Not used; percentages are passed through directly

        else:
            raise ValueError(f"Unknown tolerance_mode: {self.tolerance_mode!r}")

    def _pct_to_absolute_tolerance(self, pct_tolerances, y_true, value_range):
        """Convert percentage tolerances to absolute tolerance values.

        Parameters
        ----------
        pct_tolerances : ndarray
            Tolerance percentages in the sweep grid.
        y_true : ndarray
            Ground truth values (unused for range/absolute modes).
        value_range : float or None
            Pre-computed value range (``None`` for absolute mode).

        Returns
        -------
        ndarray
            Absolute tolerance values, same shape as ``pct_tolerances``.
        """
        if self.tolerance_mode in ("range", "quantile_range"):
            # tolerance = (pct × range) / 2
            # The division by 2 ensures 100% tolerance = ±50% of range,
            # mirroring common visual "band" interpretations.
            return (pct_tolerances / 100.0) * value_range / 2.0

        elif self.tolerance_mode == "absolute":
            # pct values are treated as raw absolute tolerances
            return pct_tolerances

        else:
            raise ValueError(f"Unknown tolerance_mode: {self.tolerance_mode!r}")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {}  # default: range mode, fine step
        params2 = {"tolerance_mode": "quantile_range", "step_pct": 1.0}
        params3 = {"tolerance_mode": "absolute", "max_tolerance_pct": 10.0}
        params4 = {"tolerance_mode": "relative", "step_pct": 1.0}

        return [params1, params2, params3, params4]
