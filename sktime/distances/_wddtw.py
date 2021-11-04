# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

from typing import Union

import numpy as np
from numba import njit

from sktime.distances._ddtw import DerivativeCallable, _average_of_slope
from sktime.distances._numba_utils import is_no_python_compiled_callable
from sktime.distances._squared import _SquaredDistance
from sktime.distances._wdtw import _wdtw_numba_distance
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
        custom_distance: DistanceCallable = _SquaredDistance().distance_factory,
        bounding_matrix: np.ndarray = None,
        compute_derivative: DerivativeCallable = _average_of_slope,
        g: float = 0.0,
        **kwargs: dict,
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
        compute_derivative: Callable[[np.ndarray], np.ndarray],
                                defaults = average slope difference (see above)
            Callable that computes the derivative. If none is provided the average of
            the slope between two points used.
        g: float, defaults = 0.
            Constant that controls the curvature (slope) of the function; that is, g
            controls the level of penalisation for the points with larger phase
            difference.
        kwargs: dict
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
        """
        _bounding_matrix = resolve_bounding_matrix(
            x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
        )

        if not is_no_python_compiled_callable(compute_derivative):
            raise (
                f"The derivative callable must be no_python compiled. The name"
                f"of the callable that must be compiled is "
                f"{compute_derivative.__name__}"
            )

        # This needs to be here as potential distances only known at runtime not
        # compile time so having this at the top would cause circular import errors.
        from sktime.distances._distance import distance_factory

        _custom_distance = distance_factory(x, y, metric=custom_distance, **kwargs)

        @njit()
        def numba_wddtw_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            _x = compute_derivative(_x)
            _y = compute_derivative(_y)
            return _wdtw_numba_distance(_x, _y, _custom_distance, _bounding_matrix, g)

        return numba_wddtw_distance
