# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

from typing import Callable, Union

import numpy as np
from numba import njit

from sktime.distances._numba_utils import _compute_pairwise_distance
from sktime.distances._squared import _SquaredDistance
from sktime.distances.base import DistanceCallable, NumbaDistance
from sktime.distances.lower_bounding import LowerBounding, resolve_bounding_matrix


class _EdrDistance(NumbaDistance):
    """Edit distance for real sequences (edr) between two timeseries."""

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
        window: int = 2,
        itakura_max_slope: float = 2.0,
        custom_distance: DistanceCallable = _SquaredDistance().distance_factory,
        bounding_matrix: np.ndarray = None,
        epsilon: float = 0.5,
        **kwargs: dict
    ) -> DistanceCallable:
        """Create a no_python compiled edr distance callable.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First timeseries.
        y: np.ndarray (2d array)
            Second timeseries.
        lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
            Lower bounding technique to use.
            If LowerBounding enum provided, the following are valid:
                LowerBounding.NO_BOUNDING - No bounding
                LowerBounding.SAKOE_CHIBA - Sakoe chiba
                LowerBounding.ITAKURA_PARALLELOGRAM - Itakura parallelogram
            If int value provided, the following are valid:
                1 - No bounding
                2 - Sakoe chiba
                3 - Itakura parallelogram
        window: int, defaults = 2
            Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding).
        itakura_max_slope: float, defaults = 2.
            Gradient of the slope for itakura parallelogram (if using Itakura
            Parallelogram lower bounding).
        custom_distance: str or Callable, defaults = squared
            The distance metric to use.
            If a string is given, see sktime/distances/distance/_distance.py for a
            list of valid string values.

            If callable then it has to be a distance factory or numba distance callable.
            If you want to pass custom kwargs to the distance at runtime, use a distance
            factory as it constructs the distance before distance computation.
            A distance callable takes the form (must be no_python compiled):
            Callable[
                [np.ndarray, np.ndarray],
                float
            ]

            A distance factory takes the form (must return a no_python callable):
            Callable[
                [np.ndarray, np.ndarray, bool, dict],
                Callable[[np.ndarray, np.ndarray], float]
            ]
        bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y))
            Custom bounding matrix to use. If defined then other lower bounding params
            are ignored. The matrix should be structure so that indexes
            considered in bound are the value 0. and indexes outside the bounding
            matrix should be infinity.
        epsilon : float, defaults = 0.5
            Matching threshold to determine if distance between two subsequences are
            considered similar (similar if distance less than the threshold).
        kwargs: dict
            Extra arguments for custom distances. See the documentation for the
            distance itself for valid kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled edr distance callable.

        Raises
        ------
        ValueError
            If the input timeseries is not a numpy array.
            If the input timeseries doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
            If epsilon is not a float.
        """
        _bounding_matrix = resolve_bounding_matrix(
            x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
        )

        if not isinstance(epsilon, float):
            raise ValueError("The value of epsilon must be a float.")

        # This needs to be here as potential distances only known at runtime not
        # compile time so having this at the top would cause circular import errors.
        from sktime.distances._distance import distance_factory

        _custom_distance = distance_factory(x, y, metric=custom_distance, **kwargs)

        @njit(fastmath=True)
        def numba_edr_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            return _numba_edr_distance(
                _x, _y, _custom_distance, _bounding_matrix, epsilon
            )

        return numba_edr_distance


@njit(cache=True, fastmath=True)
def _numba_edr_distance(
    x: np.ndarray,
    y: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float],
    bounding_matrix: np.ndarray,
    epsilon: float,
) -> float:
    """Edr distance compiled to no_python.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First timeseries.
    y: np.ndarray (2d array)
        Second timeseries.
    distance: Callable[[np.ndarray, np.ndarray], float],
                    defaults = squared_distance
        Distance function to used to compute distance between timeseries.
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Bounding matrix where the values in bound are marked by finite values and
        outside bound points are infinite values.
    epsilon : float
        Matching threshold to determine if distance between two subsequences are
        considered similar (similar if distance less than the threshold).

    Returns
    -------
    float
        Edr between two timeseries.
    """
    symmetric = np.array_equal(x, y)
    pre_computed_distances = _compute_pairwise_distance(x, y, symmetric, distance)

    cost_matrix = _edr_cost_matrix(
        x, y, bounding_matrix, pre_computed_distances, epsilon
    )

    return float(cost_matrix[-1, -1] / max(x.shape[0], y.shape[0]))


@njit(cache=True, fastmath=True)
def _edr_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    pre_computed_distances: np.ndarray,
    epsilon: float,
):
    """Compute the edr cost matrix between two timeseries.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First timeseries.
    y: np.ndarray (2d array)
        Second timeseries.
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Bounding matrix where the values in bound are marked by finite values and
        outside bound points are infinite values.
    pre_computed_distances: np.ndarray (2d of size mxn where m is len(x) and n is
                                        len(y))
        Pre-computed distances.
    epsilon : float
        Matching threshold to determine if distance between two subsequences are
        considered similar (similar if distance less than the threshold).

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Edr cost matrix between x and y.
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if np.isfinite(bounding_matrix[i - 1, j - 1]):
                if pre_computed_distances[i - 1, j - 1] < epsilon:
                    cost = 0
                else:
                    cost = 1
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1] + cost,
                    cost_matrix[i - 1, j] + 1,
                    cost_matrix[i, j - 1] + 1,
                )
    return cost_matrix[1:, 1:]
