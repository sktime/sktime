import numpy as np
from numba import njit
from typing import Tuple, Callable

from sktime.distances.distance_rework._ddtw import average_of_slope
from sktime.distances.distance_rework._wdtw import _WdtwDistance
from sktime.distances.distance_rework.base import DistanceCostCallable
from sktime.distances.distance_rework.base._types import DerivativeCallable


class _WddtwDistance(_WdtwDistance):

    def independent_distance_factory(
            self,
            x: np.ndarray,
            y: np.ndarray,
            return_cost_matrix: bool = False,
            **kwargs: dict
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        """Create a no_python distance for independent distance.

        This method will take only a univariate time series and compute the distance.

        Parameters
        ----------
        x: np.ndarray (1d array of shape (m1,))
            First time series
        y: np.ndarray (1d array of shape (m2,))
            Second time series
        return_cost_matrix: bool, default = False
            Boolean that when true will also return the cost matrix.
        kwargs: dict
            kwargs for the given distance metric

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            Callable where two, numpy 2d arrays are taken as parameters (x and y),
            a cost matrix used to compute the distance is returned and a float
            that represents the dependent distance between x and y.
        """
        if 'compute_derivative' in kwargs:
            compute_derivative = kwargs['compute_derivative']
        else:
            compute_derivative = average_of_slope

        x = compute_derivative(x)
        y = compute_derivative(y)
        wdtw_callable = super()._independent_distance_factory(x, y, **kwargs)

        if return_cost_matrix is True:
            # Return callable that sums the cost matrix
            @njit()
            def _distance_callable(
                    _x: np.ndarray, _y: np.ndarray
            ) -> Tuple[np.ndarray, float]:
                _x = compute_derivative(_x)
                _y = compute_derivative(_y)
                total = 0
                cost_matrix = np.zeros((_x.shape[1], _y.shape[1]))
                if _x.ndim > 1:
                    for i in range(x.shape[0]):
                        curr_cost_matrix, curr_dist = wdtw_callable(_x[i], _y[i])
                        cost_matrix = np.add(cost_matrix, curr_cost_matrix)
                        total += curr_dist
                else:
                    cost_matrix, total = wdtw_callable(_x, _y)
                return cost_matrix, total

        else:
            @njit()
            def _distance_callable(_x: np.ndarray, _y: np.ndarray) -> float:
                _x = compute_derivative(_x)
                _y = compute_derivative(_y)
                total = 0
                if _x.ndim > 1:
                    for i in range(x.shape[0]):
                        _, curr_dist = wdtw_callable(_x[i], _y[i])
                        total += curr_dist
                else:
                    _, total = wdtw_callable(_x, _y)
                return total

        return _distance_callable

    def _dependent_distance_factory(
            self,
            x: np.ndarray,
            y: np.ndarray,
            window: int = None,
            itakura_max_slope: float = None,
            bounding_matrix: np.ndarray = None,
            compute_derivative: DerivativeCallable = average_of_slope,
            g: float = 0.0,
            **kwargs: dict,
    ) -> DistanceCostCallable:
        format_kwargs = {
            "window": window,
            "itakura_max_slope": itakura_max_slope,
            "bounding_matrix": bounding_matrix,
            "compute_derivative": compute_derivative,
            "g": g
        }
        format_kwargs = {**format_kwargs, **kwargs}

        x = compute_derivative(x)
        y = compute_derivative(y)
        wdtw_callable = super()._dependent_distance_factory(x, y, **format_kwargs)

        @njit()
        def _wddtw_distance(
                _x: np.ndarray,
                _y: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
            _x = compute_derivative(_x)
            _y = compute_derivative(_y)
            return wdtw_callable(_x, _y)

        return _wddtw_distance
