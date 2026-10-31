"""Regression tests for constrained Time Warp Edit distances."""

import numpy as np
import pytest

from sktime.dists_kernels._numba_distances import twe_alignment_path, twe_distance
from sktime.dists_kernels._numba_distances._twe_numba import _twe_cost_matrix


@pytest.mark.parametrize("n_dimensions", [1, 2])
def test_twe_diagonal_bounding_matrix(n_dimensions):
    """A diagonal constraint admits only the sum of diagonal matching costs."""
    x = np.tile(np.arange(1.0, 9.0), (n_dimensions, 1))
    y = np.full_like(x, 10.0)
    bounding_matrix = np.full((9, 9), np.inf)
    np.fill_diagonal(bounding_matrix, 0.0)

    cost_matrix = _twe_cost_matrix(x, y, bounding_matrix, 1.0, 0.001, 2)

    # 每步匹配同时计入当前点和前一个点的距离; 原点由补零得到。
    point_costs = np.linalg.norm(x - y, axis=0)
    expected = np.cumsum(point_costs + np.r_[0.0, point_costs[:-1]])
    np.testing.assert_allclose(np.diag(cost_matrix)[1:], expected)
    assert cost_matrix[0, 0] == 0.0
    assert np.isinf(cost_matrix[~np.eye(9, dtype=bool)]).all()


@pytest.mark.parametrize("window", [0.0, 0.1])
def test_twe_distance_diagonal_window(window):
    """Narrow windows cannot start a path from an unvisited zero-cost cell."""
    x = np.arange(1.0, 9.0)
    y = np.full(8, 10.0)

    assert twe_distance(x, y, window=window) == pytest.approx(86.0)
    path, distance, cost_matrix = twe_alignment_path(
        x, y, window=window, return_cost_matrix=True
    )
    assert distance == pytest.approx(86.0)
    assert path == [(i, i) for i in range(8)]
    assert cost_matrix.shape == (9, 9)


def test_twe_unreachable_endpoint():
    """A constraint with no complete path has infinite cost."""
    x = np.array([[1.0, 2.0, 3.0]])
    bounding_matrix = np.zeros((4, 4))
    bounding_matrix[-1, -1] = np.inf
    cost_matrix = _twe_cost_matrix(x, x, bounding_matrix, 1.0, 0.001, 2)
    assert np.isinf(cost_matrix[-1, -1])


def test_twe_widening_window():
    """Admitting more paths cannot increase the minimum accumulated cost."""
    x = np.arange(1.0, 9.0)
    y = np.full(8, 10.0)
    distances = [twe_distance(x, y, window=w) for w in [0.0, 0.25, 1.0]]
    assert distances[0] >= distances[1] >= distances[2]
