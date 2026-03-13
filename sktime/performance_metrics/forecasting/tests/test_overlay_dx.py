"""Tests for OverlayDX metric, including relative tolerance mode."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Utkarshkarki"]

import numpy as np
import pandas as pd
import pytest

from sktime.performance_metrics.forecasting import OverlayDX, overlay_dx_score
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(OverlayDX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
class TestOverlayDX:
    """Test suite for OverlayDX metric (all tolerance modes)."""

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_perfect_prediction(self):
        """Perfect predictions must yield score = 1.0."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0, 2.0])
        y_pred = y_true.copy()

        metric = OverlayDX()
        score = metric(y_true, y_pred)

        assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    def test_completely_off_prediction(self):
        """Very poor predictions must yield a very low score."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        metric = OverlayDX()
        score = metric(y_true, y_pred)

        assert score < 0.1, f"Expected very low score, got {score}"

    def test_partial_accuracy(self):
        """Partial accuracy should produce a score in (0, 1)."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0, 2.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0, 1.25])

        metric = OverlayDX()
        score = metric(y_true, y_pred)

        assert 0.0 < score < 1.0, f"Expected score in (0,1), got {score}"
        assert score > 0.5, f"Expected decent score for small errors, got {score}"

    def test_score_in_unit_interval(self):
        """Score must always lie in [0, 1] for all modes."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, -1.0])
        y_pred = np.array([1.1, 1.9, 3.5, 4.0, 5.5, 8.0, -0.5])

        for mode in ["range", "quantile_range", "absolute", "relative"]:
            kwargs = (
                {"max_tolerance_pct": 10.0} if mode == "absolute" else {}
            )
            metric = OverlayDX(tolerance_mode=mode, step_pct=1.0, **kwargs)
            score = metric(y_true, y_pred)
            assert 0.0 <= score <= 1.0, (
                f"Score out of [0,1] for mode={mode!r}: {score}"
            )

    # ------------------------------------------------------------------
    # Constant series edge cases
    # ------------------------------------------------------------------

    def test_constant_series_perfect(self):
        """Constant true series with perfect prediction → 1.0."""
        y_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        score = OverlayDX()(y_true, y_pred)
        assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    def test_constant_series_imperfect(self):
        """Constant true series with any error → 0.0."""
        y_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 6.0, 5.5, 4.5, 5.2])

        score = OverlayDX()(y_true, y_pred)
        assert np.isclose(score, 0.0), f"Expected 0.0, got {score}"

    # ------------------------------------------------------------------
    # Tolerance modes – global (O(N log N + K))
    # ------------------------------------------------------------------

    def test_tolerance_mode_range(self):
        """range mode produces a valid score."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        score = OverlayDX(tolerance_mode="range")(y_true, y_pred)
        assert 0.0 < score < 1.0

    def test_tolerance_mode_quantile_range(self):
        """quantile_range mode produces a valid score."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # outlier
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 100.1])

        score_r = OverlayDX(tolerance_mode="range")(y_true, y_pred)
        score_q = OverlayDX(tolerance_mode="quantile_range")(y_true, y_pred)

        assert 0.0 < score_r < 1.0
        assert 0.0 < score_q < 1.0

    def test_tolerance_mode_absolute(self):
        """absolute mode produces a valid score."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([12.0, 22.0, 32.0, 42.0, 52.0])  # constant +2 error

        score = OverlayDX(
            tolerance_mode="absolute",
            max_tolerance_pct=10.0,
            min_tolerance_pct=0.1,
            step_pct=0.5,
        )(y_true, y_pred)
        assert 0.0 < score < 1.0

    def test_quantile_range_small_sample_warning(self):
        """quantile_range with n < 20 falls back to range with UserWarning."""
        y_true = np.array([1.0, 2.0, 3.0])  # n=3
        y_pred = np.array([1.1, 2.1, 3.1])

        with pytest.warns(UserWarning, match="small for quantile_range"):
            score = OverlayDX(tolerance_mode="quantile_range")(y_true, y_pred)

        assert 0.0 <= score <= 1.0

    # ------------------------------------------------------------------
    # Relative tolerance mode (O(N × K)) – new in this PR
    # ------------------------------------------------------------------

    def test_relative_mode_perfect_prediction(self):
        """Relative mode: perfect prediction → score = 1.0."""
        y_true = np.array([3.0, 5.0, 10.0, 20.0, 50.0])
        y_pred = y_true.copy()

        score = OverlayDX(tolerance_mode="relative")(y_true, y_pred)
        assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    def test_relative_mode_valid_score_range(self):
        """Relative mode score must lie in [0, 1]."""
        y_true = np.array([1.0, 10.0, 100.0, 1000.0])
        y_pred = np.array([1.05, 10.5, 102.0, 1010.0])

        score = OverlayDX(tolerance_mode="relative", step_pct=1.0)(y_true, y_pred)
        assert 0.0 <= score <= 1.0, f"Score out of [0,1]: {score}"

    def test_relative_mode_scale_invariance(self):
        """Relative mode is scale-invariant; scaling y_true/y_pred by constant
        must not change the score."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 3.3, 4.4, 5.5])  # 10% error everywhere

        metric = OverlayDX(tolerance_mode="relative", step_pct=1.0)

        score_orig = metric(y_true, y_pred)
        score_scaled = metric(y_true * 100, y_pred * 100)

        assert np.isclose(score_orig, score_scaled, rtol=1e-6), (
            f"Scores differ after scaling: {score_orig} vs {score_scaled}"
        )

    def test_relative_mode_proportional_error(self):
        """Relative mode: uniform p% error on all points should yield the same
        score regardless of absolute scale."""
        # 5% relative error at every point
        y_true = np.array([10.0, 20.0, 50.0, 100.0])
        y_pred = y_true * 1.05

        score = OverlayDX(
            tolerance_mode="relative",
            max_tolerance_pct=100.0,
            min_tolerance_pct=0.1,
            step_pct=0.1,
        )(y_true, y_pred)

        # At τ ≥ 5%, all points are covered → coverage jumps to 100% at τ=5%
        # Score should be noticeably > 0 and < 1
        assert 0.0 < score < 1.0, f"Unexpected score for 5% uniform error: {score}"

    def test_relative_mode_vs_range_mode_differ(self):
        """Relative and range modes should in general return different scores
        for heterogeneous series (different scales)."""
        # Series with very different magnitudes: relative errors are uniform
        # but absolute errors are not → modes must differ
        y_true = np.array([1.0, 100.0, 10000.0])
        y_pred = y_true * 1.1  # 10% relative error everywhere

        score_range = OverlayDX(
            tolerance_mode="range", step_pct=1.0
        )(y_true, y_pred)
        score_rel = OverlayDX(
            tolerance_mode="relative", step_pct=1.0
        )(y_true, y_pred)

        # They should not be (nearly) identical
        assert not np.isclose(score_range, score_rel, rtol=1e-3), (
            "Expected different scores for range vs relative modes "
            f"on heterogeneous series: range={score_range}, rel={score_rel}"
        )

    def test_relative_mode_zero_true_values(self):
        """Relative mode: points with y_true=0 have zero tolerance;
        covered only when error is exactly 0."""
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.0, 1.0, 2.0])  # perfect match

        score = OverlayDX(tolerance_mode="relative", step_pct=1.0)(y_true, y_pred)
        assert np.isclose(score, 1.0), f"Perfect match should give 1.0, got {score}"

        # Non-zero error on the zero-true point
        y_pred_err = np.array([0.5, 1.0, 2.0])  # only first point wrong
        score_err = OverlayDX(tolerance_mode="relative", step_pct=1.0)(
            y_true, y_pred_err
        )
        # Score must be < 1 (one point is never covered)
        assert score_err < 1.0, f"Expected score < 1.0, got {score_err}"

    def test_relative_mode_with_pandas_series(self):
        """Relative mode works with pandas Series inputs."""
        y_true = pd.Series([1.0, 5.0, 10.0, 20.0])
        y_pred = pd.Series([1.05, 5.25, 10.5, 21.0])  # 5% relative error

        score = OverlayDX(tolerance_mode="relative", step_pct=1.0)(y_true, y_pred)
        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0

    # ------------------------------------------------------------------
    # Multivariate / multioutput
    # ------------------------------------------------------------------

    def test_multioutput_uniform_average(self):
        """Multivariate with uniform averaging returns a scalar in [0, 1]."""
        y_true = pd.DataFrame({"v1": [3.0, -0.5, 2.0], "v2": [1.0, 2.0, 3.0]})
        y_pred = pd.DataFrame({"v1": [2.5, 0.0, 2.0], "v2": [1.1, 2.2, 2.9]})

        score = OverlayDX(multioutput="uniform_average")(y_true, y_pred)
        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0

    def test_multioutput_raw_values(self):
        """Multivariate with raw_values returns one score per variable."""
        y_true = pd.DataFrame({"v1": [3.0, -0.5, 2.0], "v2": [1.0, 2.0, 3.0]})
        y_pred = pd.DataFrame({"v1": [2.5, 0.0, 2.0], "v2": [1.1, 2.2, 2.9]})

        scores = OverlayDX(multioutput="raw_values")(y_true, y_pred)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (2,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_multioutput_relative_mode(self):
        """Relative mode works correctly for multivariate data."""
        y_true = pd.DataFrame({"v1": [1.0, 2.0, 3.0], "v2": [10.0, 20.0, 30.0]})
        y_pred = pd.DataFrame(
            {"v1": [1.05, 2.1, 3.15], "v2": [10.5, 21.0, 31.5]}
        )  # 5% relative error in both columns

        metric = OverlayDX(
            tolerance_mode="relative", step_pct=1.0, multioutput="raw_values"
        )
        scores = metric(y_true, y_pred)

        # Both columns have identical relative error → identical scores
        assert np.isclose(scores[0], scores[1], rtol=1e-6), (
            f"Scores should be equal for identical relative errors: {scores}"
        )

    # ------------------------------------------------------------------
    # Function API
    # ------------------------------------------------------------------

    def test_function_api_range(self):
        """Function API returns valid score for default (range) mode."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0, 2.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0, 1.25])

        score = overlay_dx_score(y_true, y_pred)
        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0

    def test_function_api_relative(self):
        """Function API routes to relative mode correctly."""
        y_true = np.array([1.0, 10.0, 100.0])
        y_pred = np.array([1.05, 10.5, 105.0])

        score = overlay_dx_score(y_true, y_pred, tolerance_mode="relative")
        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0

    def test_function_api_sample_weight_raises(self):
        """Function API raises NotImplementedError for sample_weight."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        with pytest.raises(NotImplementedError, match="does not support sample_weight"):
            overlay_dx_score(y_true, y_pred, sample_weight=np.ones(3))

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def test_invalid_tolerance_mode(self):
        """Unsupported tolerance_mode must raise ValueError."""
        with pytest.raises(ValueError, match="tolerance_mode must be one of"):
            OverlayDX(tolerance_mode="invalid_mode")

    def test_max_less_than_min(self):
        """max_tolerance_pct <= min_tolerance_pct must raise ValueError."""
        with pytest.raises(ValueError, match="max_tolerance_pct.*must be >"):
            OverlayDX(max_tolerance_pct=1.0, min_tolerance_pct=10.0)

    def test_negative_min_tolerance(self):
        """min_tolerance_pct <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="min_tolerance_pct must be > 0"):
            OverlayDX(min_tolerance_pct=-1.0)

    def test_invalid_step_zero(self):
        """step_pct = 0 must raise ValueError."""
        with pytest.raises(ValueError, match="step_pct must be"):
            OverlayDX(step_pct=0.0)

    def test_invalid_step_too_large(self):
        """step_pct > (max - min) must raise ValueError."""
        with pytest.raises(ValueError, match="step_pct must be"):
            OverlayDX(max_tolerance_pct=10.0, min_tolerance_pct=1.0, step_pct=20.0)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_arrays_raise(self):
        """Empty input arrays must raise ValueError."""
        with pytest.raises(ValueError, match="empty arrays"):
            OverlayDX()(np.array([]), np.array([]))

    def test_nan_in_input_raises(self):
        """NaN in inputs must raise ValueError."""
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        with pytest.raises(ValueError, match="NaN"):
            OverlayDX()(y_true, y_pred)

    def test_inf_in_input_raises(self):
        """Inf in inputs must raise ValueError."""
        y_true = np.array([1.0, np.inf, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        with pytest.raises(ValueError, match="Inf"):
            OverlayDX()(y_true, y_pred)

    def test_single_point_perfect(self):
        """Single-point perfect prediction (constant series) → 1.0."""
        score = OverlayDX()(np.array([5.0]), np.array([5.0]))
        assert np.isclose(score, 1.0)

    def test_single_point_imperfect(self):
        """Single-point with error (constant series) → 0.0."""
        score = OverlayDX()(np.array([5.0]), np.array([6.0]))
        assert np.isclose(score, 0.0)

    def test_relative_single_point_perfect(self):
        """Relative mode, single-point perfect → 1.0."""
        score = OverlayDX(tolerance_mode="relative", step_pct=1.0)(
            np.array([5.0]), np.array([5.0])
        )
        assert np.isclose(score, 1.0)

    # ------------------------------------------------------------------
    # Determinism and repr
    # ------------------------------------------------------------------

    def test_deterministic(self):
        """Metric is deterministic: same inputs yield identical scores."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        metric = OverlayDX()
        assert metric(y_true, y_pred) == metric(y_true, y_pred)

    def test_deterministic_relative(self):
        """Relative mode is also deterministic."""
        y_true = np.array([1.0, 10.0, 100.0])
        y_pred = np.array([1.1, 10.5, 99.0])

        metric = OverlayDX(tolerance_mode="relative", step_pct=1.0)
        assert metric(y_true, y_pred) == metric(y_true, y_pred)

    def test_repr_includes_parameters(self):
        """__repr__ must include all key parameters."""
        metric = OverlayDX(
            tolerance_mode="relative",
            max_tolerance_pct=50.0,
            min_tolerance_pct=1.0,
            step_pct=0.5,
        )
        r = repr(metric)
        assert "tolerance_mode" in r
        assert "relative" in r
        assert "step_pct" in r
        assert "0.5" in r

    # ------------------------------------------------------------------
    # lower_is_better tag
    # ------------------------------------------------------------------

    def test_lower_is_better_false(self):
        """OverlayDX is a score → lower_is_better must be False."""
        assert OverlayDX().get_tag("lower_is_better") is False

    # ------------------------------------------------------------------
    # get_test_params
    # ------------------------------------------------------------------

    def test_get_test_params(self):
        """get_test_params returns valid parameter sets including relative mode."""
        params_list = OverlayDX.get_test_params()
        assert isinstance(params_list, list)
        assert len(params_list) >= 1

        modes_covered = {p.get("tolerance_mode", "range") for p in params_list}
        assert "relative" in modes_covered, (
            "get_test_params should include a relative-mode parameter set"
        )

        for params in params_list:
            metric = OverlayDX(**params)
            assert isinstance(metric, OverlayDX)

    # ------------------------------------------------------------------
    # Performance characteristic: relative mode O(N×K) vs global O(N log N + K)
    # ------------------------------------------------------------------

    def test_relative_mode_complexity_characteristic(self):
        """Relative mode must accept large inputs without error (smoke test).

        With default step_pct=0.1, K≈1000 and N=100 → 100×1000=100,000 ops.
        This should complete in reasonable time.
        """
        rng = np.random.default_rng(42)
        y_true = rng.uniform(1.0, 100.0, size=100)
        y_pred = y_true * rng.uniform(0.95, 1.05, size=100)

        score = OverlayDX(tolerance_mode="relative")(y_true, y_pred)
        assert 0.0 <= score <= 1.0
