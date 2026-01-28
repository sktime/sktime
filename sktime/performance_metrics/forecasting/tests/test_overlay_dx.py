"""Tests for OverlayDX metric."""

__author__ = ["sktime developers"]

import numpy as np
import pandas as pd
import pytest

from sktime.performance_metrics.forecasting import OverlayDX, overlay_dx_score
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(OverlayDX),
    reason="skip test for OverlayDX if softdeps not available",
)
class TestOverlayDX:
    """Test OverlayDX metric class."""

    def test_perfect_prediction(self):
        """Test that perfect predictions yield score = 1.0."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0, 2.0])
        y_pred = y_true.copy()

        metric = OverlayDX()
        score = metric(y_true, y_pred)

        assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    def test_completely_off_prediction(self):
        """Test that very poor predictions yield low score."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([100.0, 200.0, 300.0, 400.0, 500.0])  # Far off

        metric = OverlayDX()
        score = metric(y_true, y_pred)

        assert score < 0.1, f"Expected very low score, got {score}"

    def test_partial_accuracy(self):
        """Test partially accurate predictions."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0, 2.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0, 1.25])

        metric = OverlayDX()
        score = metric(y_true, y_pred)

        assert 0.0 < score < 1.0, f"Expected score in (0,1), got {score}"
        assert score > 0.5, f"Expected decent score for small errors, got {score}"

    def test_constant_series_perfect(self):
        """Test constant true series with perfect prediction."""
        y_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        metric = OverlayDX()
        score = metric(y_true, y_pred)

        assert np.isclose(
            score, 1.0
        ), f"Perfect match on constant series should be 1.0, got {score}"

    def test_constant_series_imperfect(self):
        """Test constant true series with errors."""
        y_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 6.0, 5.5, 4.5, 5.2])

        metric = OverlayDX()
        score = metric(y_true, y_pred)

        assert np.isclose(
            score, 0.0
        ), f"Any error on constant series should be 0.0, got {score}"

    def test_multioutput_uniform_average(self):
        """Test multivariate with uniform averaging."""
        y_true = pd.DataFrame({"var1": [3, -0.5, 2], "var2": [1, 2, 3]})
        y_pred = pd.DataFrame({"var1": [2.5, 0.0, 2], "var2": [1.1, 2.2, 2.9]})

        metric = OverlayDX(multioutput="uniform_average")
        score = metric(y_true, y_pred)

        assert isinstance(score, (float, np.floating)), "Should return scalar"
        assert 0.0 <= score <= 1.0, f"Score should be in [0,1], got {score}"

    def test_multioutput_raw_values(self):
        """Test multivariate with per-variable scores."""
        y_true = pd.DataFrame({"var1": [3, -0.5, 2], "var2": [1, 2, 3]})
        y_pred = pd.DataFrame({"var1": [2.5, 0.0, 2], "var2": [1.1, 2.2, 2.9]})

        metric = OverlayDX(multioutput="raw_values")
        scores = metric(y_true, y_pred)

        assert isinstance(scores, np.ndarray), "Should return array"
        assert scores.shape == (2,), f"Should have 2 scores, got {scores.shape}"
        assert np.all(
            (scores >= 0) & (scores <= 1)
        ), f"Scores should be in [0,1], got {scores}"

    def test_tolerance_mode_range(self):
        """Test range-based tolerance mode."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        metric = OverlayDX(tolerance_mode="range")
        score = metric(y_true, y_pred)

        assert 0.0 < score < 1.0

    def test_tolerance_mode_quantile_range(self):
        """Test quantile-range tolerance mode (robust to outliers)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Outlier at end
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 100.1])

        metric_range = OverlayDX(tolerance_mode="range")
        metric_quantile = OverlayDX(tolerance_mode="quantile_range")

        score_range = metric_range(y_true, y_pred)
        score_quantile = metric_quantile(y_true, y_pred)

        # Both should return valid scores
        assert 0.0 < score_range < 1.0
        assert 0.0 < score_quantile < 1.0

    def test_tolerance_mode_absolute(self):
        """Test absolute tolerance mode."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([12.0, 22.0, 32.0, 42.0, 52.0])  # Constant +2 error

        # With absolute tolerance, we interpret percentages as absolute values
        metric = OverlayDX(
            tolerance_mode="absolute",
            max_tolerance_pct=10.0,  # Tolerance up to 10 units
            min_tolerance_pct=0.1,
            step_pct=0.5,
        )
        score = metric(y_true, y_pred)

        assert 0.0 < score < 1.0

    def test_parameter_validation_max_less_than_min(self):
        """Test that max_tolerance_pct > min_tolerance_pct is enforced."""
        with pytest.raises(ValueError, match="max_tolerance_pct.*must be >"):
            OverlayDX(max_tolerance_pct=1.0, min_tolerance_pct=10.0)

    def test_parameter_validation_negative_min(self):
        """Test that min_tolerance_pct > 0 is enforced."""
        with pytest.raises(ValueError, match="min_tolerance_pct must be > 0"):
            OverlayDX(min_tolerance_pct=-1.0)

    def test_parameter_validation_invalid_step(self):
        """Test that step_pct is valid."""
        with pytest.raises(ValueError, match="step_pct must be"):
            OverlayDX(step_pct=0.0)

        with pytest.raises(ValueError, match="step_pct must be"):
            OverlayDX(max_tolerance_pct=10.0, min_tolerance_pct=1.0, step_pct=20.0)

    def test_parameter_validation_invalid_tolerance_mode(self):
        """Test that tolerance_mode is one of allowed values."""
        with pytest.raises(ValueError, match="tolerance_mode must be one of"):
            OverlayDX(tolerance_mode="invalid_mode")

    def test_parameter_validation_relative_mode_not_supported(self):
        """Test that relative mode raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="'relative' mode is not supported"):
            OverlayDX(tolerance_mode="relative")

    def test_step_size_sensitivity(self):
        """Test that different step sizes yield similar but not identical scores."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        metric_fine = OverlayDX(step_pct=0.1)
        metric_coarse = OverlayDX(step_pct=5.0)

        score_fine = metric_fine(y_true, y_pred)
        score_coarse = metric_coarse(y_true, y_pred)

        # Scores should be in the same ballpark
        assert np.isclose(
            score_fine, score_coarse, rtol=0.1
        ), "Different step sizes should yield similar scores"

    def test_function_api(self):
        """Test function-based API."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0, 2.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0, 1.25])

        score = overlay_dx_score(y_true, y_pred)

        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0

    def test_function_api_sample_weight_not_supported(self):
        """Test that sample_weight raises NotImplementedError."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0, 2.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0, 1.25])
        sample_weight = np.array([1, 1, 1, 1, 1])

        with pytest.raises(NotImplementedError, match="does not support sample_weight"):
            overlay_dx_score(y_true, y_pred, sample_weight=sample_weight)

    def test_lower_is_better_tag(self):
        """Test that metric has correct lower_is_better tag."""
        metric = OverlayDX()
        assert (
            metric.get_tag("lower_is_better") is False
        ), "OverlayDX is a score, so lower_is_better should be False"

    def test_deterministic(self):
        """Test that metric is deterministic."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        metric = OverlayDX()

        score1 = metric(y_true, y_pred)
        score2 = metric(y_true, y_pred)

        assert score1 == score2, "Metric should be deterministic"

    def test_repr_includes_parameters(self):
        """Test that __repr__ includes key parameters."""
        metric = OverlayDX(
            tolerance_mode="quantile_range",
            max_tolerance_pct=50.0,
            min_tolerance_pct=1.0,
            step_pct=0.5,
        )

        repr_str = repr(metric)

        assert "tolerance_mode" in repr_str
        assert "quantile_range" in repr_str
        assert "step_pct" in repr_str
        assert "0.5" in repr_str

    def test_get_test_params(self):
        """Test that get_test_params returns valid parameter sets."""
        params_list = OverlayDX.get_test_params()

        assert isinstance(params_list, list)
        assert len(params_list) >= 1

        # Test that each parameter set creates a valid instance
        for params in params_list:
            metric = OverlayDX(**params)
            assert isinstance(metric, OverlayDX)

    def test_with_pandas_series(self):
        """Test that metric works with pandas Series."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])

        metric = OverlayDX()
        score = metric(y_true, y_pred)

        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0

    def test_with_pandas_dataframe(self):
        """Test that metric works with pandas DataFrame."""
        y_true = pd.DataFrame({"var1": [1.0, 2.0, 3.0], "var2": [4.0, 5.0, 6.0]})
        y_pred = pd.DataFrame({"var1": [1.1, 2.1, 3.1], "var2": [4.1, 5.1, 6.1]})

        metric = OverlayDX(multioutput="uniform_average")
        score = metric(y_true, y_pred)

        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0

    def test_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        y_true = np.array([])
        y_pred = np.array([])

        metric = OverlayDX()
        with pytest.raises(ValueError, match="empty arrays"):
            metric(y_true, y_pred)

    def test_nan_in_input(self):
        """Test that NaN values raise ValueError."""
        y_true = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        metric = OverlayDX()
        with pytest.raises(ValueError, match="NaN"):
            metric(y_true, y_pred)

    def test_inf_in_input(self):
        """Test that Inf values raise ValueError."""
        y_true = np.array([1.0, 2.0, 3.0, np.inf, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        metric = OverlayDX()
        with pytest.raises(ValueError, match="Inf"):
            metric(y_true, y_pred)

    def test_quantile_range_small_sample(self):
        """Test that small samples fall back to range mode with warning."""
        y_true = np.array([1.0, 2.0, 3.0])  # n=3 < 20
        y_pred = np.array([1.1, 2.1, 3.1])

        metric = OverlayDX(tolerance_mode="quantile_range")

        # Should issue warning but still work
        with pytest.warns(UserWarning, match="small for quantile_range"):
            score = metric(y_true, y_pred)

        assert 0.0 <= score <= 1.0

    def test_single_point_perfect(self):
        """Test that single point predictions work correctly."""
        y_true = np.array([5.0])
        y_pred = np.array([5.0])

        metric = OverlayDX()
        score = metric(y_true, y_pred)

        # Single point, perfect match should be 1.0
        # (constant series with perfect prediction)
        assert np.isclose(score, 1.0)

    def test_single_point_imperfect(self):
        """Test that single point with error works correctly."""
        y_true = np.array([5.0])
        y_pred = np.array([6.0])

        metric = OverlayDX()
        score = metric(y_true, y_pred)

        # Single point constant series with error should be 0.0
        assert np.isclose(score, 0.0)
