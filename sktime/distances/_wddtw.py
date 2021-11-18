# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

from typing import Any, Union

import numpy as np
from numba import njit

from sktime.distances._ddtw import DerivativeCallable, _average_of_slope
from sktime.distances._numba_utils import is_no_python_compiled_callable
from sktime.distances._wdtw import _weighted_cost_matrix
from sktime.distances.base import DistanceCallable, NumbaDistance
from sktime.distances.lower_bounding import LowerBounding, resolve_bounding_matrix


class _WddtwDistance(NumbaDistance):
    """Weighted derivative dynamic time warping (wddtw) distance between two series."""

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
        window: int = 2,
        itakura_max_slope: float = 2.0,
        bounding_matrix: np.ndarray = None,
        compute_derivative: DerivativeCallable = _average_of_slope,
        g: float = 0.0,
        **kwargs: Any,
    ) -> DistanceCallable:
        """Create a no_python compiled wddtw distance callable.

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
        bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y)),
                                        defaults = None)
            Custom bounding matrix to use. If defined then other lower_bounding params
            are ignored. The matrix should be structure so that indexes considered in
            bound should be the value 0. and indexes outside the bounding matrix should
            be infinity.
        compute_derivative: Callable[[np.ndarray], np.ndarray],
                                defaults = average slope difference
            Callable that computes the derivative. If none is provided the average of
            the slope between two points used.
        g: float, defaults = 0.
            Constant that controls the curvature (slope) of the function; that is, g
            controls the level of penalisation for the points with larger phase
            difference.
        kwargs: Any
            Extra arguments for custom distances. See the documentation for the
            distance itself for valid kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled wddtw distance callable.

        Raises
        ------
        ValueError
            If the input timeseries is not a numpy array.
            If the input timeseries doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
            If the compute derivative callable is not no_python compiled.
            If the value of g is not a float
        """
        _bounding_matrix = resolve_bounding_matrix(
            x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
        )

        if not isinstance(g, float):
            raise ValueError(
                f"The value of g must be a float. The current value is {g}"
            )

        if not is_no_python_compiled_callable(compute_derivative):
            raise ValueError(
                f"The derivative callable must be no_python compiled. The name"
                f"of the callable that must be compiled is "
                f"{compute_derivative.__name__}"
            )

        @njit()
        def numba_wddtw_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            _x = compute_derivative(_x)
            _y = compute_derivative(_y)
            cost_matrix = _weighted_cost_matrix(_x, _y, _bounding_matrix, g)
            return cost_matrix[-1, -1]

        return numba_wddtw_distance
