__author__ = ["chrisholder", "TonyBagnall"]


from typing import Any, Callable

import numpy as np

from sktime.distances.base import (
    DistanceAlignmentPathCallable,
    DistanceCallable,
    NumbaDistance,
)

DerivativeCallable = Callable[[np.ndarray], np.ndarray]


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
        The derivative of the time series X.
    """
    from sktime.distances._ddtw_numba import average_of_slope

    derivative_X = []
    for val in X:
        derivative_X.append(average_of_slope(val))
    return np.array(derivative_X)


class _DdtwDistance(NumbaDistance):
    """Derivative dynamic time warping (ddtw) between two time series.

    Takes the slope based derivative of the series (using compute_derivative), then
    applies DTW (using the _cost_matrix from _DtwDistance)
    """

    def _distance_alignment_path_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        compute_derivative=None,
        **kwargs: Any,
    ) -> DistanceAlignmentPathCallable:
        """Create a no_python compiled ddtw distance alignment path callable.

        Series should be shape (d, m), where d is the number of dimensions, m the series
        length. Series can be different lengths.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        return_cost_matrix: bool, defaults = False
            Boolean that when true will also return the cost matrix.
        window: float, defaults = None
            Float that is the radius of the Sakoe-Chiba window (if using Sakoe-Chiba
            lower bounding). Must be between 0 and 1.
        itakura_max_slope: float, defaults = None
            Gradient of the slope for Itakura parallelogram (if using Itakura
            Parallelogram lower bounding). Must be between 0 and 1.
        bounding_matrix: np.ndarray (2d array of shape (m1,m2)), defaults = None
            Custom bounding matrix to use. If defined then other lower_bounding params
            are ignored. The matrix should be structure so that indexes considered in
            bound should be the value 0. and indexes outside the bounding matrix should
            be infinity.
        compute_derivative: Callable[[np.ndarray], np.ndarray],
                                defaults = average slope difference
            Callable that computes the derivative. If none is provided the average of
            the slope between two points used.
        kwargs: any
            extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float]]
            No_python compiled wdtw distance path callable.

        Raises
        ------
        ValueError
            If the input time series is not a numpy array.
            If the input time series doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
            If the compute derivative callable is not no_python compiled.
        """
        from sktime.distances._distance_alignment_paths import compute_min_return_path
        from sktime.distances._dtw_numba import _cost_matrix
        from sktime.distances._numba_utils import is_no_python_compiled_callable
        from sktime.distances.lower_bounding import resolve_bounding_matrix
        from sktime.utils.numba.njit import njit

        if compute_derivative is None:
            from sktime.distances._ddtw_numba import average_of_slope

            compute_derivative = average_of_slope

        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        if not is_no_python_compiled_callable(compute_derivative):
            raise TypeError(
                f"The derivative callable must be no_python compiled. The name"
                f"of the callable that must be compiled is "
                f"{compute_derivative.__name__}"
            )

        if return_cost_matrix is True:

            @njit(cache=True)
            def numba_ddtw_distance_alignment_path(
                _x: np.ndarray,
                _y: np.ndarray,
            ) -> tuple[list, float, np.ndarray]:
                _x = compute_derivative(_x)
                _y = compute_derivative(_y)
                cost_matrix = _cost_matrix(_x, _y, _bounding_matrix)
                path = compute_min_return_path(cost_matrix, _bounding_matrix)
                return path, cost_matrix[-1, -1], cost_matrix

        else:

            @njit(cache=True)
            def numba_ddtw_distance_alignment_path(
                _x: np.ndarray,
                _y: np.ndarray,
            ) -> tuple[list, float]:
                _x = compute_derivative(_x)
                _y = compute_derivative(_y)
                cost_matrix = _cost_matrix(_x, _y, _bounding_matrix)
                path = compute_min_return_path(cost_matrix, _bounding_matrix)
                return path, cost_matrix[-1, -1]

        return numba_ddtw_distance_alignment_path

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        compute_derivative=None,
        **kwargs: Any,
    ) -> DistanceCallable:
        """Create a no_python compiled ddtw distance callable.

        Series should be shape (d, m), where d is the number of dimensions, m the series
        length. Series can be different lengths.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        window: float, defaults = None
            Float that is the radius of the Sakoe-Chiba window (if using Sakoe-Chiba
            lower bounding). Must be between 0 and 1.
        itakura_max_slope: float, defaults = None
            Gradient of the slope for Itakura parallelogram (if using Itakura
            Parallelogram lower bounding). Must be between 0 and 1.
        bounding_matrix: np.ndarray (2d array of shape (m1,m2)), defaults = None
            Custom bounding matrix to use. If defined then other lower_bounding params
            are ignored. The matrix should be structure so that indexes considered in
            bound should be the value 0. and indexes outside the bounding matrix should
            be infinity.
        compute_derivative: Callable[[np.ndarray], np.ndarray],
                                defaults = average slope difference
            Callable that computes the derivative. If none is provided the average of
            the slope between two points used.
        kwargs: any
            extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled ddtw distance callable.

        Raises
        ------
        ValueError
            If the input time series is not a numpy array.
            If the input time series doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
            If the compute derivative callable is not no_python compiled.
        """
        from sktime.distances._dtw_numba import _cost_matrix
        from sktime.distances._numba_utils import is_no_python_compiled_callable
        from sktime.distances.lower_bounding import resolve_bounding_matrix
        from sktime.utils.numba.njit import njit

        if compute_derivative is None:
            from sktime.distances._ddtw_numba import average_of_slope

            compute_derivative = average_of_slope

        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        if not is_no_python_compiled_callable(compute_derivative):
            raise TypeError(
                f"The derivative callable must be no_python compiled. The name"
                f"of the callable that must be compiled is "
                f"{compute_derivative.__name__}"
            )

        @njit(cache=True)
        def numba_ddtw_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            _x = compute_derivative(_x)
            _y = compute_derivative(_y)
            cost_matrix = _cost_matrix(_x, _y, _bounding_matrix)
            return cost_matrix[-1, -1]

        return numba_ddtw_distance
