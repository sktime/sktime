# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

from typing import Callable, Union

import numpy as np
from numba import njit

from sktime.dists_kernels.numba.distances._numba_utils import _compute_pairwise_distance
from sktime.dists_kernels.numba.distances._squared_distance import _SquaredDistance
from sktime.dists_kernels.numba.distances.base import DistanceCallable, NumbaDistance
from sktime.dists_kernels.numba.distances.dtw_based.lower_bounding import (
    LowerBounding,
    resolve_bounding_matrix,
)


class _DtwDistance(NumbaDistance):
    """Dynamic time warping (DTW) between two timeseries."""

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
        window: int = 2,
        itakura_max_slope: float = 2.0,
        custom_distance: DistanceCallable = _SquaredDistance().distance_factory,
        bounding_matrix: np.ndarray = None,
        **kwargs: dict
    ) -> DistanceCallable:
        """Create a no_python compiled dtw distance callable.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First timeseries.
        y: np.ndarray (2d array)
            Second timeseries.
        lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
            Lower bounding technique to use.
        window: int, defaults = 2
            Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding).
        itakura_max_slope: float, defaults = 2.
            Gradient of the slope for itakura parallelogram (if using Itakura
            Parallelogram lower bounding).
        custom_distance: Callable[[np.ndarray, np.ndarray], float],
                        defaults = squared_distance
            Distance function to used to compute distance between aligned timeseries.
        bounding_matrix: np.ndarray (2d array)
            Custom bounding matrix to use. If defined then other lower_bounding params
            and creation are ignored. The matrix should be structure so that indexes
            considered in bound should be the value 0. and indexes outside the bounding
            matrix should be infinity.
        kwargs: dict
            Extra arguments for custom distance should be put in the kwargs. See the
            documentation for the distance for kwargs.


        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled Dtw distance callable.

        Raises
        ------
        ValueError
            If the input timeseries is not a numpy array.
            If the input timeseries doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
        """
        _bounding_matrix = resolve_bounding_matrix(
            x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
        )

        # This needs to be here as potential distances only known at runtime not
        # compile time so having this at the top would cause circular import errors.
        from sktime.dists_kernels.numba.distances.distance import distance_factory

        _custom_distance = distance_factory(x, y, metric=custom_distance, **kwargs)

        @njit()
        def numba_dtw_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            return _dtw_numba_distance(_x, _y, _custom_distance, _bounding_matrix)

        return numba_dtw_distance


@njit(cache=True)
def _dtw_numba_distance(
    x: np.ndarray,
    y: np.ndarray,
    custom_distance: Callable[[np.ndarray, np.ndarray], float],
    bounding_matrix: np.ndarray,
) -> float:
    """Dtw distance compiled to no_python.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First timeseries.
    y: np.ndarray (2d array)
        Second timeseries.
    custom_distance: Callable[[np.ndarray, np.ndarray], float],
        Distance function to used to compute distance between timeseries.
    bounding_matrix: np.ndarray (2d array)
        Bounding matrix to restrict the warping path between the two timeseries.
        The matrix should be structure so that indexes considered in bound should be
        the value 0. and indexes outside the bounding matrix should be infinity

    Returns
    -------
    distance: float
        Dtw distance between the two timeseries.
    """
    symmetric = np.array_equal(x, y)
    pre_computed_distances = _compute_pairwise_distance(
        x, y, symmetric, custom_distance
    )

    cost_matrix = _cost_matrix(x, y, bounding_matrix, pre_computed_distances)
    return cost_matrix[-1, -1]


@njit(cache=True)
def _cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    pre_computed_distances: np.ndarray,
):
    """Compute the dtw cost matrix between two timeseries.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First timeseries.
    y: np.ndarray (2d array)
        Second timeseries.
    bounding_matrix: np.ndarray (2d array)
        Bounding matrix to restrict the warping path between the two timeseries.
        The matrix should be structure so that indexes considered in bound should be
        the value 0. and indexes outside the bounding matrix should be infinity.
    pre_computed_distances: np.ndarray (2d of size mxn where m is len(x) and n is
                                        len(y))
        Precomputed pairwise matrix between the two timeseries.

    Returns
    -------
    np.ndarray (2d array of size mxn where m is len(x) and n is len(y))
        Cost matrix between two timeseries.
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i + 1, j + 1] = pre_computed_distances[i, j] + min(
                    cost_matrix[i, j + 1], cost_matrix[i + 1, j], cost_matrix[i, j]
                )

    return cost_matrix[1:, 1:]
