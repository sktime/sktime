# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

from typing import Union

import numpy as np
from numba import njit

from sktime.dists_kernels.numba.distances._numba_utils import (
    is_no_python_compiled_callable,
)
from sktime.dists_kernels.numba.distances._squared_distance import _SquaredDistance
from sktime.dists_kernels.numba.distances.base import DistanceCallable, NumbaDistance
from sktime.dists_kernels.numba.distances.dtw_based._ddtw_distance import (
    DerivativeCallable,
    _average_of_slope,
)
from sktime.dists_kernels.numba.distances.dtw_based._wdtw_distance import (
    _wdtw_numba_distance,
)
from sktime.dists_kernels.numba.distances.dtw_based.lower_bounding import (
    LowerBounding,
    resolve_bounding_matrix,
)


class _WddtwDistance(NumbaDistance):
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
        window: int, defaults = 2
            Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding).
        itakura_max_slope: float, defaults = 2.
            Gradient of the slope for itakura parallelogram (if using Itakura
            Parallelogram lower bounding).
        custom_distance: Callable[[np.ndarray, np.ndarray], float],
                        defaults = squared_distance
            Distance function to used to compute distance between timeseries.
        bounding_matrix: np.ndarray (2d array)
            Custom bounding matrix to use. If defined then other lower_bounding params
            and creation are ignored. The matrix should be structure so that indexes
            considered in bound should be the value 0. and indexes outside the bounding
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
            Extra arguments for custom distance should be put in the kwargs. See the
            documentation for the distance for kwargs.

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
        from sktime.dists_kernels.numba.distances.distance import distance_factory

        _custom_distance = distance_factory(x, y, metric=custom_distance, **kwargs)

        @njit()
        def numba_dtw_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            _x = compute_derivative(_x)
            _y = compute_derivative(_y)
            return _wdtw_numba_distance(_x, _y, _custom_distance, _bounding_matrix, g)

        return numba_dtw_distance
