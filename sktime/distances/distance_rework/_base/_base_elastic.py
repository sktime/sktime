from typing import Callable, Tuple, List, Union
from abc import ABC, abstractmethod
import numpy as np
from numba import njit

from sktime.distances.distance_rework._base._base import (
    BaseDistance,
    LocalDistanceCallable
)
from sktime.distances.lower_bounding import resolve_bounding_matrix

AlignmentPathCallableReturn = Union[
    List[Tuple],  # Just path
    Tuple[List[Tuple], float],  # Path and distance
    Tuple[List[Tuple], np.ndarray],  # Path and cost matrix
    Tuple[List[Tuple], float, np.ndarray]  # Path, distance and cost matrix
]
AlignmentPathCallable = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], AlignmentPathCallableReturn
]


class ElasticDistance(BaseDistance, ABC):
    _has_cost_matrix = True

    def alignment_path(
            self,
            x: np.ndarray,
            y: np.ndarray,
            strategy: str,
            return_cost_matrix: bool = False,
            return_distance: bool = False,
            **kwargs: dict
    ) -> AlignmentPathCallableReturn:
        """Alignment path between two time series."""
        return self.alignment_path_factory(
            x,
            y,
            strategy=strategy,
            return_distance=return_distance,
            return_cost_matrix=return_cost_matrix,
            **kwargs
        )(x, y)

    def independent_alignment_path(
            self,
            x: np.ndarray,
            y: np.ndarray,
            return_distance: bool = False,
            return_cost_matrix: bool = False,
            **kwargs: dict
    ) -> AlignmentPathCallableReturn:
        """Independent alignment path between two time series."""
        return self.alignment_path(
            x,
            y,
            strategy="independent",
            return_distance=return_distance,
            return_cost_matrix=return_cost_matrix,
            **kwargs
        )

    def dependent_alignment_path(
            self,
            x: np.ndarray,
            y: np.ndarray,
            return_distance: bool = False,
            return_cost_matrix: bool = False,
            **kwargs: dict
    ) -> AlignmentPathCallableReturn:
        """Dependent alignment path between two time series."""
        return self.alignment_path(
            x,
            y,
            strategy="dependent",
            return_distance=return_distance,
            return_cost_matrix=return_cost_matrix,
            **kwargs
        )

    def alignment_path_factory(
            self,
            x: np.ndarray,
            y: np.ndarray,
            strategy: str,
            return_cost_matrix: bool = False,
            return_distance: bool = False,
            **kwargs: dict
    ) -> AlignmentPathCallable:
        distance_callable = self.distance_factory(
            x, y, strategy=strategy, return_cost_matrix=True
        )

        alignment_callable = self._alignment_path_factory(
            x, y, strategy=strategy, **kwargs
        )
        bounding_matrix = self._get_bounding_matrix(x, y, **kwargs)

        if self._numba_distance is True:
            _alignment_callable = njit(
                cache=self._cache, fastmath=self._fastmath
            )(alignment_callable)
        else:
            _alignment_callable = alignment_callable

        _preprocess_time_series = self._preprocess_time_series_factory(x, y, **kwargs)

        if return_distance and return_cost_matrix:
            def _alignment(_x: np.ndarray, _y: np.ndarray):
                distance, cost_matrix = distance_callable(_x, _y)
                temp_x = _preprocess_time_series(_x)
                temp_y = _preprocess_time_series(_y)
                alignment = _alignment_callable(
                    temp_x, temp_y, cost_matrix, bounding_matrix
                )
                return alignment, distance, cost_matrix
        elif return_distance:
            def _alignment(_x: np.ndarray, _y: np.ndarray):
                distance, cost_matrix = distance_callable(_x, _y)
                temp_x = _preprocess_time_series(_x)
                temp_y = _preprocess_time_series(_y)
                alignment = _alignment_callable(
                    temp_x, temp_y, cost_matrix, bounding_matrix
                )
                return alignment, distance
        elif return_cost_matrix:
            def _alignment(_x: np.ndarray, _y: np.ndarray):
                distance, cost_matrix = distance_callable(_x, _y)
                temp_x = _preprocess_time_series(_x)
                temp_y = _preprocess_time_series(_y)
                alignment = _alignment_callable(
                    temp_x, temp_y, cost_matrix, bounding_matrix
                )
                return alignment, cost_matrix
        else:
            def _alignment(_x: np.ndarray, _y: np.ndarray):
                distance, cost_matrix = distance_callable(_x, _y)
                temp_x = _preprocess_time_series(_x)
                temp_y = _preprocess_time_series(_y)
                alignment = _alignment_callable(
                    temp_x, temp_y, cost_matrix, bounding_matrix
                )
                return alignment

        if self._numba_distance is True:
            return njit(cache=self._cache, fastmath=self._fastmath)(_alignment)

        return _alignment

    def _local_distance(self, x: float, y: float,
                        **kwargs: dict) -> LocalDistanceCallable:
        raise ValueError("Local distance not implemented for elastic distances.")

    def _get_bounding_matrix(
            self,
            x: np.ndarray,
            y: np.ndarray,
            window: float = None,
            itakura_max_slope: float = None,
            bounding_matrix: np.ndarray = None,
            **kwargs
    ):
        _preprocess_time_series = self._preprocess_time_series_factory(x, y, **kwargs)
        return resolve_bounding_matrix(
            _preprocess_time_series(x),
            _preprocess_time_series(y),
            window,
            itakura_max_slope,
            bounding_matrix
        )

    def _alignment_path_factory(
            self, x: np.ndarray, y: np.ndarray, strategy: str = 'independent', **kwargs
    ) -> AlignmentPathCallable:
        def _compute_min_return_path(
                _x: np.ndarray,
                _y: np.ndarray,
                _cost_matrix: np.ndarray,
                _bounding_matrix: np.ndarray
        ) -> List[Tuple]:
            x_size = _x.shape[-1]
            y_size = _y.shape[-1]

            for i in range(x_size):
                for j in range(y_size):
                    if not np.isfinite(_bounding_matrix[i, j]):
                        _cost_matrix[i, j] = np.inf

            i = _cost_matrix.shape[0] - 1
            j = _cost_matrix.shape[1] - 1
            alignment = []
            while True:
                alignment.append((i, j))

                if alignment[-1] == (0, 0):
                    break

                arr = np.array(
                    [
                        _cost_matrix[i - 1, j - 1],
                        _cost_matrix[i - 1, j],
                        _cost_matrix[i, j - 1],
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

        return _compute_min_return_path
