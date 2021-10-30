# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

from typing import Callable, Union

import numpy as np
from numba import njit

from sktime.dists_kernels.numba.distances._numba_utils import _compute_pairwise_distance
from sktime.dists_kernels.numba.distances._squared_distance import _SquaredDistance
from sktime.dists_kernels.numba.distances.base import DistanceCallable, NumbaDistance
from sktime.dists_kernels.numba.distances.dtw_based.lower_bounding import LowerBounding


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

        _bounding_matrix = _DtwDistance._resolve_bounding_matrix(
            x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
        )

        # This needs to be here as potential distances only known at runtime not
        # compile time so having this at the top would produce errors (circular import).
        from sktime.dists_kernels.numba.distances.distance import distance_factory

        _custom_distance = distance_factory(x, y, metric=custom_distance)

        @njit()
        def numba_dtw_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            return _dtw_numba_distance(_x, _y, _custom_distance, _bounding_matrix)

        return numba_dtw_distance

    @staticmethod
    def _resolve_bounding_matrix(
        x: np.ndarray,
        y: np.ndarray,
        lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
        window: int = 2,
        itakura_max_slope: float = 2.0,
        bounding_matrix: np.ndarray = None,
    ):
        """Resolve the bounding matrix parameters."""
        if bounding_matrix is None:
            if isinstance(lower_bounding, int):
                lower_bounding = LowerBounding(lower_bounding)
            else:
                lower_bounding = lower_bounding

            return lower_bounding.create_bounding_matrix(
                x,
                y,
                sakoe_chiba_window_radius=window,
                itakura_max_slope=itakura_max_slope,
            )
        else:
            return bounding_matrix


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
    return np.sqrt(cost_matrix[-1, -1])


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
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Bounding matrix where the values inside the bound are finite values (0s) and
        outside the bounds are infinity (non finite).
    pre_computed_distances: np.ndarray (2d of size mxn where m is len(x) and n is
                                        len(y))
        Precomputed pairwise matrix between two timeseries.
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
