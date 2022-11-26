# -*- coding: utf-8 -*-
"""Base class for elastic distance functions."""
from typing import Callable, List, Tuple, Union

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
AlignmentPathCallableReturn = Union[
    List[Tuple],  # Just path
    Tuple[List[Tuple], float],  # Path and distance
    Tuple[List[Tuple], np.ndarray],  # Path and cost matrix
    Tuple[List[Tuple], float, np.ndarray],  # Path, distance and cost matrix
]
AlignmentPathCallable = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], AlignmentPathCallableReturn
]


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
    def _independent_factory(distance_callable: Callable) -> Callable:
        """Create independent callable that has a cost matrix

        Parameters
        ----------
        distance_callable: Callable
            Distance callable that returns (distance, cost matrix)

        Returns
        -------
        Callable
            Independent distance callable.
        """

        def _distance_callable(_x: np.ndarray, _y: np.ndarray, *args):
            distance = 0
            cost_matrix = np.zeros((_x.shape[-1], _y.shape[-1]))
            for i in range(len(_x)):
                curr_dist, curr_cost_matrix = distance_callable(_x[i], _y[i], *args)
                cost_matrix = np.add(cost_matrix, curr_cost_matrix)
                distance += curr_dist
            return distance, cost_matrix

        return _distance_callable

    def alignment_path_factory(
        self,
        strategy: str,
        return_cost_matrix: bool = False,
        return_distance: bool = False,
    ):
        """Create an alignment path callable.

        Parameters
        ----------
        strategy : str
            Strategy to use for distance calculation. Either "dependent" or
            "independent".
        return_cost_matrix : bool, default = False
            Whether to return the cost matrix.
        return_distance : bool, default = False
            Whether to return the distance.

        Returns
        -------
        Callable
            Alignment path callable.
        """

        if self._return_cost_matrix is False:
            raise ValueError("Distance does not support alignment path.")

        dist_callable = self.distance_factory(
            strategy=strategy, return_cost_matrix=True
        )
        alignment_path_callable = self._convert_to_numba(self._alignment_path)
        preprocess_ts = self._preprocess_ts_factory(strategy="dependent")

        def alignment_path_all(x, y, *args):
            _x = preprocess_ts(x, *args)
            _y = preprocess_ts(y, *args)
            distance, cost_matrix = dist_callable(x, y, *args)
            bounding_matrix = get_bounding_matrix(_x, _y, *args)
            path = alignment_path_callable(x, y, cost_matrix, bounding_matrix, *args)
            return path, distance, cost_matrix

        alignment_path = self._convert_to_numba(alignment_path_all)

        if not return_cost_matrix and return_distance:

            def _alginment_path(x, y, *args):
                path, distance, cost_matrix = alignment_path(x, y, *args)
                return path, distance

            return self._convert_to_numba(_alginment_path)

        if return_cost_matrix and not return_distance:

            def _alginment_path(x, y, *args):
                path, distance, cost_matrix = alignment_path(x, y, *args)
                return path, cost_matrix

            return self._convert_to_numba(_alginment_path)

        if not return_cost_matrix and not return_distance:

            def _alginment_path(x, y, *args):
                path, distance, cost_matrix = alignment_path(x, y, *args)
                return path

            return self._convert_to_numba(_alginment_path)

        return alignment_path

    def alignment_path(
        self,
        x: np.ndarray,
        y: np.ndarray,
        strategy: str,
        return_cost_matrix: bool = False,
        return_distance: bool = False,
        *args
    ) -> AlignmentPathCallableReturn:
        """Alignment path between two time series.

        Parameters
        ----------
        x: np.ndarray
            First time series.
        y: np.ndarray
            Second time series.
        return_cost_matrix: bool, defualt = False
            Whether to return the cost matrix.
        return_distance: bool, default = False
            Whether to return the distance.

        Returns
        -------
        list of tuples
            Alignment path.
        float (optional if return_distance is true)
            Distance between x and y.
        np.ndarray (optional if return_cost_matrix is true)
            Cost matrix.
        """
        return self.alignment_path_factory(
            strategy=strategy,
            return_distance=return_distance,
            return_cost_matrix=return_cost_matrix,
        )(x, y, *args)

    @staticmethod
    def _alignment_path(
        x: np.ndarray, y: np.ndarray, cost_matrix: np.ndarray, bounding_matrix, *args
    ) -> List[Tuple]:
        x_size = x.shape[-1]
        y_size = y.shape[-1]

        for i in range(x_size):
            for j in range(y_size):
                if not np.isfinite(bounding_matrix[i, j]):
                    cost_matrix[i, j] = np.inf

        i = cost_matrix.shape[0] - 1
        j = cost_matrix.shape[1] - 1
        alignment = []
        while True:
            alignment.append((i, j))

            if alignment[-1] == (0, 0):
                break

            arr = np.array(
                [
                    cost_matrix[i - 1, j - 1],
                    cost_matrix[i - 1, j],
                    cost_matrix[i, j - 1],
                ]
            )
            min_index = np.argmin(arr)

            if min_index == 0:
                i = i - 1
                j = j - 1
            elif min_index == 1:
                i = i - 1
            else:
                j = j - 1

        return alignment[::-1]


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
