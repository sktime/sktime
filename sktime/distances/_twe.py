# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Any, List, Tuple

import numpy as np

from sktime.distances.base import DistanceCallable, NumbaDistance


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
        from sktime.distances._distance_alignment_paths import compute_twe_return_path
        from sktime.distances._twe_numba import _twe_cost_matrix
        from sktime.distances.lower_bounding import resolve_bounding_matrix
        from sktime.utils.numba.njit import njit

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
        from sktime.distances._twe_numba import _twe_cost_matrix, pad_ts
        from sktime.distances.lower_bounding import resolve_bounding_matrix
        from sktime.utils.numba.njit import njit

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
