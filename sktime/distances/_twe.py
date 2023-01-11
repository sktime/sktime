# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

import warnings
from typing import Any, List, Tuple

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from sktime.distances._distance_alignment_paths import compute_twe_return_path
from sktime.distances.base import DistanceCallable, NumbaDistance
from sktime.distances.lower_bounding import resolve_bounding_matrix

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)


class _TweDistance(NumbaDistance):
    """Time Warp Edit (TWE) distance between two time series.

    The Time Warp Edit (TWE) distance is a distance measure for discrete time series
    matching with time 'elasticity'. In comparison to other distance measures, (e.g.
    DTW (Dynamic Time Warping) or LCS (Longest Common Subsequence Problem)), TWE is a
    metric. Its computational time complexity is O(n^2), but can be drastically reduced
    in some specific situation by using a corridor to reduce the search space. Its
    memory space complexity can be reduced to O(n). It was first proposed in [1].

    References
    ----------
    .. [1] Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment
    for Time Series Matching". IEEE Transactions on Pattern Analysis and Machine
    Intelligence. 31 (2): 306–318.
    """

    def _distance_alignment_path_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        lmbda: float = 1.0,
        nu: float = 0.001,
        p: int = 2,
        **kwargs: Any
    ) -> DistanceCallable:
        """Create a no_python compiled twe distance callable.

        Series should be shape (d, m), where d is the number of dimensions, m the series
        length.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        return_cost_matrix: bool, defaults = False
            Boolean that when true will also return the cost matrix.
        window: Float, defaults = None
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
        lmbda: float, defaults = 1.0
            A constant penalty that punishes the editing efforts. Must be >= 1.0.
        nu: float, defaults = 0.001
            A non-negative constant which characterizes the stiffness of the elastic
            twe measure. Must be > 0.
        p: int, defaults = 2
            Order of the p-norm for local cost.
        kwargs: any
            extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled Dtw distance callable.

        Raises
        ------
        ValueError
            If the input time series are not numpy array.
            If the input time series do not have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
        """
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        if return_cost_matrix is True:

            @njit(cache=True)
            def numba_twe_distance_alignment_path(
                _x: np.ndarray,
                _y: np.ndarray,
            ) -> Tuple[List, float, np.ndarray]:
                cost_matrix = _twe_cost_matrix(_x, _y, _bounding_matrix, lmbda, nu, p)
                path = compute_twe_return_path(cost_matrix, _bounding_matrix)
                return path, cost_matrix[-1, -1], cost_matrix

        else:

            @njit(cache=True)
            def numba_twe_distance_alignment_path(
                _x: np.ndarray,
                _y: np.ndarray,
            ) -> Tuple[List, float]:
                cost_matrix = _twe_cost_matrix(_x, _y, _bounding_matrix, lmbda, nu, p)
                path = compute_twe_return_path(cost_matrix, _bounding_matrix)
                return path, cost_matrix[-1, -1]

        return numba_twe_distance_alignment_path

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        lmbda: float = 1.0,
        nu: float = 0.001,
        p: int = 2,
        **kwargs: Any
    ) -> DistanceCallable:
        """Create a no_python compiled twe distance callable.

        Series should be shape (d, m), where d is the number of dimensions, m the series
        length.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        window: Float, defaults = None
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
        lmbda: float, defaults = 1.0
            A constant penalty that punishes the editing efforts. Must be >= 1.0.
        nu: float, defaults = 0.001
            A non-negative constant which characterizes the stiffness of the elastic
            twe measure. Must be > 0.
        p: int, defaults = 2
            Order of the p-norm for local cost.
        kwargs: any
            extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled Dtw distance callable.

        Raises
        ------
        ValueError
            If the input time series are not numpy array.
            If the input time series do not have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
        """
        x = pad_ts(x)
        y = pad_ts(y)
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        @njit(cache=True)
        def numba_twe_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            cost_matrix = _twe_cost_matrix(_x, _y, _bounding_matrix, lmbda, nu, p)
            return cost_matrix[-1, -1]

        return numba_twe_distance


@njit(cache=True)
def pad_ts(x: np.ndarray) -> np.ndarray:
    """Pad the time with a 0.0 at the start.

    Parameters
    ----------
    x: np.ndarray (of shape (d, m))
        A time series.

    Returns
    -------
    np.ndarray
        A padded time series of shape (d, m + 1)
    """
    padded_x = np.zeros((x.shape[0], x.shape[1] + 1))
    zero_arr = np.array([0.0])
    for i in range(x.shape[0]):
        padded_x[i, :] = np.concatenate((zero_arr, x[i, :]))
    return padded_x


@njit(cache=True)
def _twe_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    lmbda: float,
    nu: float,
    p: int,
) -> np.ndarray:
    """Twe distance compiled to no_python.

    Series should be shape (d, m), where d is the number of dimensions, m the series
    length. Series can be different lengths.

    Parameters
    ----------
    x: np.ndarray (2d array of shape dxm1).
        First time series.
    y: np.ndarray (2d array of shape dxm1).
        Second time series.
    bounding_matrix: np.ndarray (2d array of shape m1xm2)
        Bounding matrix where the index in bound finite values (0.) and indexes
        outside bound points are infinite values (non finite).
    lmbda: float
        A constant penalty that punishes the editing efforts. Must be >= 1.0.
    nu: float
        A non-negative constant which characterizes the stiffness of the elastic
        TWE measure. Must be > 0.
    p: int
        Order of the p-norm for local cost.

    Returns
    -------
    cost_matrix: np.ndarray (of shape (n, m) where n is the len(x) and m is len(y))
        The dtw cost matrix.
    """
    x = pad_ts(x)
    y = pad_ts(y)
    x_size = x.shape[1]
    y_size = y.shape[1]
    dimensions = x.shape[0]

    cost_matrix = np.zeros((x_size, y_size))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    delete_addition = nu + lmbda

    for i in range(1, x_size):
        for j in range(1, y_size):
            if np.isfinite(bounding_matrix[i, j]):
                # Deletion in x
                # Euclidean distance to x[:, i - 1] and y[:, i]
                deletion_x_euclid_dist = 0
                for k in range(dimensions):
                    deletion_x_euclid_dist += (x[k][i - 1] - y[k][i]) ** 2
                deletion_x_euclid_dist = np.sqrt(deletion_x_euclid_dist)

                del_x = cost_matrix[i - 1, j] + deletion_x_euclid_dist + delete_addition

                # Deletion in y
                # Euclidean distance to x[:, j - 1] and y[:, j]
                deletion_y_euclid_dist = 0
                for k in range(dimensions):
                    deletion_y_euclid_dist += (x[k][j - 1] - y[k][j]) ** 2
                deletion_y_euclid_dist = np.sqrt(deletion_y_euclid_dist)

                del_y = cost_matrix[i, j - 1] + deletion_y_euclid_dist + delete_addition

                # Keep data points in both time series
                # Euclidean distance to x[:, i] and y[:, j]
                match_same_euclid_dist = 0
                for k in range(dimensions):
                    match_same_euclid_dist += (x[k][i] - y[k][j]) ** 2
                match_same_euclid_dist = np.sqrt(match_same_euclid_dist)

                # Euclidean distance to x[:, i - 1] and y[:, j - 1]
                match_previous_euclid_dist = 0
                for k in range(dimensions):
                    match_previous_euclid_dist += (x[k][i - 1] - y[k][j - 1]) ** 2
                match_previous_euclid_dist = np.sqrt(match_previous_euclid_dist)

                match = (
                    cost_matrix[i - 1, j - 1]
                    + match_same_euclid_dist
                    + match_previous_euclid_dist
                    + (nu * (2 * abs(i - j)))
                )

                # Choose the operation with the minimal cost and update DP Matrix
                cost_matrix[i, j] = min(del_x, del_y, match)
    return cost_matrix
