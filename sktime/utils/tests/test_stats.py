"""Tests for weighted statistics utilities in sktime.utils.stats.

Tests in this module
--------------------
    test_weighted_percentile_uniform_weights_matches_numpy
    test_weighted_percentile_2d_shape
    test_weighted_percentile_extreme_percentiles
    test_weighted_percentile_concentrated_weight
    test_weighted_geometric_mean_uniform_equals_scipy
    test_weighted_geometric_mean_single_element
    test_weighted_geometric_mean_incompatible_shapes
    test_weighted_geometric_mean_heavy_weight_dominates
    test_weighted_median_uniform_weights
    test_weighted_min_max_bracket_data
    test_weighted_min_max_ordering_invariant
"""

__author__ = ["oashe"]

import numpy as np
import pytest
from scipy.stats import gmean as scipy_gmean

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.stats import (
    _weighted_geometric_mean,
    _weighted_max,
    _weighted_median,
    _weighted_min,
    _weighted_percentile,
)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
class TestWeightedPercentile:
    """Tests for _weighted_percentile function."""

    def test_uniform_weights_matches_numpy_median(self):
        """With equal weights, the 50th percentile should be close to np.median."""
        rng = np.random.RandomState(0)
        array = rng.rand(101)
        weights = np.ones(101)
        result = _weighted_percentile(array, weights, percentile=50)
        np_median = np.median(array)
        # allow tolerance because _weighted_percentile uses lower-bound convention
        assert abs(result - np_median) < 0.05

    def test_2d_column_independence(self):
        """Each column of a 2D array should have its percentile computed independently."""
        array = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        weights = np.ones(3)
        result = _weighted_percentile(array, weights, percentile=50)
        assert result.shape == (2,)
        # column 0 median ~ 2, column 1 median ~ 200
        assert result[0] <= 3.0
        assert result[1] >= 100.0

    def test_concentrated_weight_returns_that_element(self):
        """If one element has overwhelming weight, percentile should return it."""
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([0.0, 0.0, 0.0, 0.0, 1000.0])
        result = _weighted_percentile(array, weights, percentile=50)
        assert result == 5.0

    @pytest.mark.parametrize("percentile", [0, 100])
    def test_extreme_percentiles(self, percentile):
        """Percentile 0 returns min, percentile 100 returns max."""
        array = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        weights = np.ones(5)
        result = _weighted_percentile(array, weights, percentile=percentile)
        if percentile == 0:
            assert result == 10.0
        else:
            assert result == 50.0


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
class TestWeightedGeometricMean:
    """Tests for _weighted_geometric_mean function."""

    def test_uniform_weights_equals_scipy_gmean(self):
        """Uniform weights should match scipy's geometric mean."""
        rng = np.random.RandomState(42)
        y = rng.rand(1, 5) + 0.1  # positive values
        weights = np.ones(5)
        result = _weighted_geometric_mean(y, weights=weights, axis=1)
        expected = scipy_gmean(y, axis=1)
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_single_element_is_identity(self):
        """Geometric mean of a single value equals that value."""
        y = np.array([[7.5]])
        weights = np.array([1.0])
        result = _weighted_geometric_mean(y, weights=weights, axis=1)
        np.testing.assert_almost_equal(result, 7.5, decimal=10)

    def test_heavy_weight_dominates(self):
        """An element with overwhelming weight should dominate the result."""
        y = np.array([[1.0, 1000.0]])
        weights = np.array([1e-10, 1.0])
        result = _weighted_geometric_mean(y, weights=weights, axis=1)
        # result should be very close to 1000
        assert result[0] > 900.0

    def test_incompatible_2d_shapes_raise(self):
        """Incompatible 2D y and weights shapes should raise ValueError."""
        y = np.array([[1.0, 2.0, 3.0]])
        weights = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="inconsistent shapes"):
            _weighted_geometric_mean(y, weights=weights, axis=1)

    def test_1d_weights_wrong_length_on_axis0(self):
        """1D weights inconsistent with axis=0 dimension should raise."""
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            _weighted_geometric_mean(y, weights=weights, axis=0)

    def test_1d_weights_wrong_length_on_axis1(self):
        """1D weights inconsistent with axis=1 dimension should raise."""
        y = np.array([[1.0, 2.0, 3.0]])
        weights = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Input features"):
            _weighted_geometric_mean(y, weights=weights, axis=1)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
class TestWeightedMedianMinMax:
    """Tests for _weighted_median, _weighted_min, _weighted_max."""

    def test_min_max_bracket_data(self):
        """Weighted min/max should bracket the data range."""
        rng = np.random.RandomState(7)
        y = rng.rand(1, 20) * 100
        weights = rng.rand(20) + 0.01
        w_min = _weighted_min(y, axis=1, weights=weights)
        w_max = _weighted_max(y, axis=1, weights=weights)
        assert w_min[0] <= y.min() + 1e-10
        assert w_max[0] >= y.max() - 1e-10

    def test_min_less_equal_median_less_equal_max(self):
        """Weighted min <= weighted median <= weighted max must always hold."""
        rng = np.random.RandomState(123)
        y = rng.rand(1, 15) * 50
        weights = np.ones(15)
        w_min = _weighted_min(y, axis=1, weights=weights)[0]
        w_med = _weighted_median(y, axis=1, weights=weights)[0]
        w_max = _weighted_max(y, axis=1, weights=weights)[0]
        assert w_min <= w_med <= w_max

    def test_uniform_data_all_equal(self):
        """For constant data, min == median == max."""
        y = np.full((1, 10), 42.0)
        weights = np.ones(10)
        w_min = _weighted_min(y, axis=1, weights=weights)[0]
        w_med = _weighted_median(y, axis=1, weights=weights)[0]
        w_max = _weighted_max(y, axis=1, weights=weights)[0]
        assert w_min == w_med == w_max == 42.0

    def test_single_element_all_agree(self):
        """Single element: min == median == max == that element."""
        y = np.array([[3.14]])
        weights = np.array([1.0])
        assert _weighted_min(y, axis=1, weights=weights)[0] == 3.14
        assert _weighted_median(y, axis=1, weights=weights)[0] == 3.14
        assert _weighted_max(y, axis=1, weights=weights)[0] == 3.14
