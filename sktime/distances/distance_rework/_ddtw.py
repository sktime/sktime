from typing import Tuple, Callable
import numpy as np
from numba import njit

from sktime.distances.distance_rework.base import DistanceCostCallable
from sktime.distances.distance_rework import _DtwDistance

from sktime.distances.distance_rework.base._types import DerivativeCallable


def average_of_slope_transform(X: np.ndarray) -> np.ndarray:
    """Compute the average of a slope between points for multiple series.

    Parameters
    ----------
    X: np.ndarray (of shape (d, m) where d is the dimensions and m is the timepoints.
        A time series.

    Returns
    -------
    np.ndarray (2d array of shape nxm where n is len(q.shape[0]-2) and m is
                len(q.shape[1]))
        The derviative of the time series X.
    """
    derivative_X = []
    for val in X:
        derivative_X.append(average_of_slope(val))
    return np.array(derivative_X)


@njit(cache=True)
def average_of_slope(q: np.ndarray) -> np.ndarray:
    r"""Compute the average of a slope between points.

    Computes the average of the slope of the line through the point in question and
    its left neighbour, and the slope of the line through the left neighbour and the
    right neighbour. proposed in [1] for use in this context.

    .. math::
    q'_(i) = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Where q is the original time series and q' is the derived time series.

    Parameters
    ----------
    q: np.ndarray (of shape (d, m) where d is the dimensions and m is the timepoints.
        A time series.

    Returns
    -------
    np.ndarray (2d array of shape nxm where n is len(q.shape[0]-2) and m is
                len(q.shape[1]))
        Array containing the derivative of q.

    References
    ----------
    .. [1] Keogh E, Pazzani M Derivative dynamic time warping. In: proceedings of 1st
    SIAM International Conference on Data Mining, 2001
    """
    q = q.transpose()
    q2 = 0.25 * q[2:] + 0.5 * q[1:-1] - 0.75 * q[:-2]
    q2 = q2.transpose()
    return q2


class _DdtwDistance(_DtwDistance):

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
        if 'window' in kwargs:
            window = kwargs['window']
        else:
            window = None
        if 'itakura_max_slope' in kwargs:
            itakura_max_slope = kwargs['itakura_max_slope']
        else:
            itakura_max_slope = None
        if 'bounding_matrix' in kwargs:
            bounding_matrix = kwargs['bounding_matrix']
        else:
            bounding_matrix = None

        x = compute_derivative(x)
        y = compute_derivative(y)

        example_x = x
        example_y = y

        if example_x.ndim > 1:
            example_x = example_x[0]
            example_y = example_y[0]

        dtw_callable = super()._independent_distance_factory(
            example_x,
            example_y,
            window=window,
            itakura_max_slope=itakura_max_slope,
            bounding_matrix=bounding_matrix,
            **kwargs
        )

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
                        curr_cost_matrix, curr_dist = dtw_callable(_x[i], _y[i])
                        cost_matrix = np.add(cost_matrix, curr_cost_matrix)
                        total += curr_dist
                else:
                    cost_matrix, total = dtw_callable(_x, _y)
                return cost_matrix, total

        else:
            @njit()
            def _distance_callable(_x: np.ndarray, _y: np.ndarray) -> float:
                _x = compute_derivative(_x)
                _y = compute_derivative(_y)
                total = 0
                if _x.ndim > 1:
                    for i in range(x.shape[0]):
                        cost_matrix, curr_dist = dtw_callable(_x[i], _y[i])
                        total += curr_dist
                else:
                    _, total = dtw_callable(_x, _y)
                return total

        return _distance_callable

    def _dependent_distance_factory(
            self,
            x: np.ndarray,
            y: np.ndarray,
            window: float = None,
            itakura_max_slope: float = None,
            bounding_matrix: np.ndarray = None,
            compute_derivative: DerivativeCallable = average_of_slope,
            **kwargs: dict
    ) -> DistanceCostCallable:
        format_kwargs = {
            "window": window,
            "itakura_max_slope": itakura_max_slope,
            "bounding_matrix": bounding_matrix,
            "compute_derivative": compute_derivative,
        }
        format_kwargs = {**format_kwargs, **kwargs}
        dtw_callable = super()._dependent_distance_factory(
            x, y, **format_kwargs
        )

        @njit('Tuple((float64[:, :], float64))(float64[:, :], float64[:, :])',
              cache=True)
        def _ddtw_distance(
                _x: np.ndarray,
                _y: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
            _x = compute_derivative(_x)
            _y = compute_derivative(_y)
            return dtw_callable(_x, _y)

        return _ddtw_distance
