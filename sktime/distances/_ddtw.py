# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

import warnings
from typing import Any, Callable

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from sktime.distances._dtw import _cost_matrix
from sktime.distances._numba_utils import is_no_python_compiled_callable
from sktime.distances.base import DistanceCallable, NumbaDistance
from sktime.distances.lower_bounding import resolve_bounding_matrix

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)

DerivativeCallable = Callable[[np.ndarray], np.ndarray]


@njit(cache=True, fastmath=True)
def _average_of_slope(q: np.ndarray):
    r"""Compute the average of a slope between points.

    Computes the average of the slope of the line through the point in question and
    its left neighbour, and the slope of the line through the left neighbour and the
    right neighbour.

    Mathematically this is defined at:

    .. math::
        D_{x}[q] = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Where q is the original timeseries and d_q is the derived timeseries.

    Parameters
    ----------
    q: np.ndarray (2d array)
        A timeseries.

    Returns
    -------
    np.ndarray (2d array of shape nxm where n is len(q.shape[0]-2) and m is
                len(q.shape[1]))
        Array containing the derivative of q.

    """
    # Taken from https://github.com/tslearn-team/tslearn/issues/180
    return 0.25 * q[2:] + 0.5 * q[1:-1] - 0.75 * q[:-2]


class _DdtwDistance(NumbaDistance):
    """Derivative dynamic time warping (ddtw) between two timeseries."""

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: int = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        compute_derivative: DerivativeCallable = _average_of_slope,
        **kwargs: Any,
    ) -> DistanceCallable:
        """Create a no_python compiled ddtw distance callable.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First timeseries.
        y: np.ndarray (2d array)
            Second timeseries.
        window: int, defaults = None
            Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding).
        itakura_max_slope: float, defaults = None
            Gradient of the slope for itakura parallelogram (if using Itakura
            Parallelogram lower bounding).
        bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y)),
                                        defaults = None
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
            If the input timeseries is not a numpy array.
            If the input timeseries doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
            If the compute derivative callable is not no_python compiled.
        """
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        if not is_no_python_compiled_callable(compute_derivative):
            raise (
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
