# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "jlines", "TonyBagnall"]

import warnings
from typing import List, Tuple

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from sktime.distances._distance_alignment_paths import compute_min_return_path
from sktime.distances.base import (
    DistanceAlignmentPathCallable,
    DistanceCallable,
    NumbaDistance,
)
from sktime.distances.lower_bounding import resolve_bounding_matrix

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)


class _MsmDistance(NumbaDistance):
    r"""Move-split-merge (MSM) distance between two time series.

    (MSM) [1] is a distance measure that is conceptually similar to other edit
    distance-based approaches, where similarity is calculated by using a set of
    operations to transform one series into another. Each operation has an
    associated cost, and three operations are defined for MSM: move, split, and merge.
    Move is synonymous with a substitution operation, where one value is replaced by
    another. Split and merge differ from other approaches, as they attempt to add
    context to insertions and deletions. The cost of inserting and deleting values
    depends on the value itself and adjacent values, rather than treating all
    insertions and deletions equally (for example, as in ERP). Therefore, the split
    operation is introduced to insert an identical copy of a value immediately after
    itself, and the merge operation is used to delete a value if it directly follows
    an identical value.

    Currently only works with univariate series.

    References
    ----------
    .. [1] Stefan A., Athitsos V., Das G.: The Move-Split-Merge metric for time
    series. IEEE Transactions on Knowledge and Data Engineering 25(6):1425–1438, 2013
    """

    def _distance_alignment_path_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        c: float = 1.0,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        **kwargs: dict,
    ) -> DistanceAlignmentPathCallable:
        """Create a no_python compiled MSM distance path callable.

        Series should be shape (1, m), where m is the series length. Series can be
        different lengths.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        return_cost_matrix: bool, defaults = False
            Boolean that when true will also return the cost matrix.
        c: float, default = 1.0
            Cost for split or merge operation.
        window: float, defaults = None
            Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding). Must be between 0 and 1.
        itakura_max_slope: float, defaults = None
            Gradient of the slope for itakura parallelogram (if using Itakura
            Parallelogram lower bounding). Must be between 0 and 1.
        bounding_matrix: np.ndarray (2d array of shape (m1,m2)), defaults = None
            Custom bounding matrix to use. If defined then other lower_bounding params
            are ignored. The matrix should be structure so that indexes considered in
            bound should be the value 0. and indexes outside the bounding matrix should
            be infinity.
        kwargs: any
            extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled MSM distance callable.

        Raises
        ------
        ValueError
            If the input time series have more than one dimension (shape[0] > 1)
            If the input time series is not a numpy array.
            If the input time series doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
            If epsilon is not a float.
        """
        if x.shape[0] > 1 or y.shape[0] > 1:
            raise ValueError(
                f"ERROR, MSM distance currently only works with "
                f"univariate series, passed seris shape {x.shape[0]} and"
                f"shape {y.shape[0]}"
            )

        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        if return_cost_matrix is True:

            @njit(cache=True)
            def numba_msm_distance_alignment_path(
                _x: np.ndarray,
                _y: np.ndarray,
            ) -> Tuple[List, float, np.ndarray]:
                cost_matrix = _cost_matrix(_x, _y, _bounding_matrix, c)
                path = compute_min_return_path(cost_matrix, _bounding_matrix)
                return path, cost_matrix[-1, -1], cost_matrix

        else:

            @njit(cache=True)
            def numba_msm_distance_alignment_path(
                _x: np.ndarray,
                _y: np.ndarray,
            ) -> Tuple[List, float]:
                cost_matrix = _cost_matrix(_x, _y, _bounding_matrix, c)
                path = compute_min_return_path(cost_matrix, _bounding_matrix)
                return path, cost_matrix[-1, -1]

        return numba_msm_distance_alignment_path

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        c: float = 1.0,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        **kwargs: dict,
    ) -> DistanceCallable:
        """Create a no_python compiled MSM distance callable.

        Series should be shape (d, m) where m is the series length. Series can be
        different lengths.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        c: float, default = 1.0
            Cost for split or merge operation.
        window: float, defaults = None
            Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding). Must be between 0 and 1.
        itakura_max_slope: float, defaults = None
            Gradient of the slope for itakura parallelogram (if using Itakura
            Parallelogram lower bounding). Must be between 0 and 1.
        bounding_matrix: np.ndarray (2d array of shape (m1,m2)), defaults = None
            Custom bounding matrix to use. If defined then other lower_bounding params
            are ignored. The matrix should be structure so that indexes considered in
            bound should be the value 0. and indexes outside the bounding matrix should
            be infinity.
        kwargs: any
            extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled MSM distance callable.

        Raises
        ------
        ValueError
            If the input time series have more than one dimension (shape[0] > 1)
            If the input time series is not a numpy array.
            If the input time series doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
            If epsilon is not a float.
        """
        if x.shape[0] > 1 or y.shape[0] > 1:
            raise ValueError(
                f"ERROR, MSM distance currently only works with "
                f"univariate series, passed seris shape {x.shape[0]} and"
                f"shape {y.shape[0]}"
            )
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        @njit(cache=True)
        def numba_msm_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            cost_matrix = _cost_matrix(_x, _y, _bounding_matrix, c)
            return cost_matrix[-1, -1]

        return numba_msm_distance


@njit(fastmath=True, cache=True)
def _cost(_x: float, _y: float, _z: float, _c: float) -> float:
    if (_y <= _x <= _z) or (_y >= _x >= _z):
        return _c
    return _c + min(abs(_x - _y), abs(_x - _z))


@njit(cache=True)
def _cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
) -> np.ndarray:
    """MSM distance compiled to no_python.

    Series should be shape (1, m), where m the series (m is currently univariate only).
    length.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series.
    y: np.ndarray (2d array)
        Second time series.
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Bounding matrix where the index in bound finite values (0.) and indexes
        outside bound points are infinite values (non finite).

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Erp cost matrix between x and y.
    """
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    # init the first cell
    if x[0][0] > y[0][0]:
        cost_matrix[0, 0] = x[0][0] - y[0][0]
    else:
        cost_matrix[0, 0] = y[0][0] - x[0][0]
    # init the rest of the first row and column
    for i in range(1, x_size):
        cost = _cost(x[0, i], x[0, i - 1], y[0, 0], c)
        cost_matrix[i][0] = cost_matrix[i - 1][0] + cost
    for i in range(1, y_size):
        cost = _cost(y[0, i], y[0, i - 1], x[0, 0], c)
        cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

    for i in range(1, x_size):
        for j in range(1, y_size):
            if np.isfinite(bounding_matrix[i, j]):
                d1 = cost_matrix[i - 1, j - 1] + np.abs(x[0][i] - y[0][j])
                d2 = cost_matrix[i - 1, j] + _cost(x[0][i], x[0][i - 1], y[0][j], c)
                d3 = cost_matrix[i, j - 1] + _cost(y[0][j], x[0][i], y[0][j - 1], c)
                cost_matrix[i][j] = min(d1, d2, d3)
    for i in range(1, x_size):
        for j in range(1, y_size):
            if np.isfinite(bounding_matrix[i, j]):
                d1 = cost_matrix[i - 1, j - 1] + abs(x[0, i] - y[0, j])
                d2 = cost_matrix[i - 1, j] + _cost(x[0, i], x[0, i - 1], y[0, j], c)
                d3 = cost_matrix[i, j - 1] + _cost(y[0, j], x[0, i], y[0, j - 1], c)
                cost_matrix[i, j] = min(d1, d2, d3)

    return cost_matrix
