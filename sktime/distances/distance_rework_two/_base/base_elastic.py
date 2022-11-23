# -*- coding: utf-8 -*-
"""Base class for elastic distance functions."""
from typing import Callable, Tuple, Union

import numpy as np
from numba import njit

from sktime.distances.distance_rework_two._base.base import (
    BaseDistance,
    DistanceCallable,
)
from sktime.distances.lower_bounding import (
    itakura_parallelogram,
    no_bounding,
    sakoe_chiba,
)

ElasticDistanceReturn = Tuple[float, np.ndarray]


class ElasticDistance(BaseDistance):
    """Base class for elastic distance."""

    _return_cost_matrix = True

    def distance_factory(
        self,
        strategy: str = "dependent",
        return_cost_matrix: bool = False,
    ) -> DistanceCallable:
        """Create an elastic distance callable.

        Parameters
        ----------
        strategy : str
            Strategy to use for distance calculation. Either "dependent" or
            "independent".
        return_cost_matrix : bool
            Whether to return the cost matrix.

        Returns
        -------
        Callable
            Distance callable.
        """
        if return_cost_matrix is True and self._return_cost_matrix is False:
            raise ValueError("Distance does not support returning cost matrix.")

        dist_callable = super().distance_factory(
            strategy=strategy,
        )

        result_distance_callable = self._elastic_result_process_factory(dist_callable)

        if return_cost_matrix:
            return result_distance_callable

        # Need to make it so it only returns the distance
        def _cm_return_dist(x, y):
            return result_distance_callable(x, y)[0]

        if self._numba_distance:
            _cm_return_dist = njit(cache=self._cache, fastmath=self._fastmath)(
                _cm_return_dist
            )

        return _cm_return_dist

    def distance(
        self, x: np.ndarray, y: np.ndarray, return_cost_matrix: bool = False, **kwargs
    ) -> Union[float, ElasticDistanceReturn]:
        """Compute the distance between two time series.

        Parameters
        ----------
        x : np.ndarray
            First time series.
        y : np.ndarray
            Second time series.
        return_cost_matrix : bool
            Whether to return the cost matrix.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        float
            Distance between x and y.
        np.ndarray (optional depending on if return_cost_matrix is true)
            Cost matrix.
        """
        strategy = kwargs.get("strategy", "dependent")
        distance_callable = self.distance_factory(
            strategy=strategy,
            return_cost_matrix=return_cost_matrix,
        )
        return distance_callable(x, y)

    def _result_process_factory(self, distance_callable: Callable):
        """Create a function to process the distance result.

        Parameters
        ----------
        distance_callable: Callable
            Distance function to process.

        Returns
        -------
        Callable
            Processed distance function.
        """
        # Just have it returning itself to ignore base class implementation
        return distance_callable

    def _elastic_result_process_factory(self, distance_callable: Callable):
        """Create a function to process the distance result for elastic distance.

        Parameters
        ----------
        distance_callable: Callable
            Distance function to process.

        Returns
        -------
        Callable
            Processed distance function.
        """
        if type(self)._result_process != BaseDistance._result_process:
            result_callback = self._convert_to_numba(self._result_process)

            def _result_process(x, y, *args):
                distance, cm = distance_callable(x, y, *args)
                return result_callback(distance, x, y, *args), cm

            return self._convert_to_numba(_result_process)

        return distance_callable

    @staticmethod
    def _independent_factory(distance_callable: Callable):
        def _distance_callable(_x: np.ndarray, _y: np.ndarray, *args):
            distance = 0
            cost_matrix = np.zeros((_x.shape[-1], _y.shape[-1]))
            for i in range(len(_x)):
                curr_dist, curr_cost_matrix = distance_callable(_x[i], _y[i], *args)
                cost_matrix = np.add(cost_matrix, curr_cost_matrix)
                distance += curr_dist
            return distance, cost_matrix

        return _distance_callable


@njit(cache=True, fastmath=True)
def get_bounding_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Union[float, None] = None,
    itakura_max_slope: Union[float, None] = None,
    bounding_matrix: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """Get bounding matrix for elastic distance.

    Parameters
    ----------
    x: np.ndarray
        First time series.
    y: np.ndarray
        Second time series.
    window: float, default=None
        Window size for Sakoe-Chiba band.
    itakura_max_slope: float, default=None
        Maximum slope for Itakura parallelogram.
    bounding_matrix: np.ndarray, default=None
        Bounding matrix to use.

    Returns
    -------
    np.ndarray
        Bounding matrix.
    """
    if bounding_matrix is not None:
        return bounding_matrix

    if itakura_max_slope is not None:
        return itakura_parallelogram(x, y, itakura_max_slope)

    if window is not None:
        return sakoe_chiba(x, y, window)

    return no_bounding(x, y)
