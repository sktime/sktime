# -*- coding: utf-8 -*-
import warnings
from typing import Union, Tuple, Callable
import numpy as np
from numba import njit

from sktime.metrics.distances.dtw._lower_bouding import LowerBounding
from sktime.metrics.distances.base.base import (
    BaseDistance,
    NumbaSupportedDistance,
    _numba_pairwise,
)
from sktime.metrics.distances._squared_dist import SquaredDistance


@njit(cache=True)
def _cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    pre_computed_distances: np.ndarray,
):
    """
    Method used to calculate the cost matrix to derive distance from

    Parameters
    ----------
    x: np.ndarray
        first time series
    y: np.ndarray
        second time series
    bounding_matrix: np.ndarray
        matrix bounding the warping path
    pre_computed_distances: np.ndarray
        pre-computed distances
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i + 1, j + 1] = pre_computed_distances[i, j] + min(
                    cost_matrix[i, j + 1], cost_matrix[i + 1, j], cost_matrix[i, j]
                )

    return cost_matrix[1:, 1:]


class Dtw(BaseDistance, NumbaSupportedDistance):
    """
    Class that is used to calculate the dynamic time warping distance

    Parameters
    ----------
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        lower bounding technique to use. Potential bounding techniques and their int
        value are given below:
        NO_BOUNDING = 1
        SAKOE_CHIBA = 2
        ITAKURA_PARALLELOGRAM = 3
    sakoe_chiba_window_radius: int, defaults = 2
        Integer that is the radius of the sakoe chiba window
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope of itakura
    custom_cost_matrix_distance: BasePairwise or Callable
        Custom pairwise distance function for distances used in dtw. If
        passing a custom Callable it takes the form: func(x, y) -> np.ndarray
        where the return is a matrix of the pairwise distance
    custom_bounding_matrix: np.ndarray, default = None
        Custom bounding matrix to use
    """

    def __init__(
        self,
        lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
        sakoe_chiba_window_radius: int = 2,
        itakura_max_slope: float = 2.0,
        custom_cost_matrix_distance: Union[BaseDistance, Callable] = None,
        custom_bounding_matrix: np.ndarray = None,
    ):
        super(Dtw, self).__init__("dtw", {"dynamic me warping"})
        self.lower_bounding: Union[LowerBounding, int] = lower_bounding
        self.sakoe_chiba_window_radius: int = sakoe_chiba_window_radius
        self.itakura_max_slope: float = itakura_max_slope

        if custom_cost_matrix_distance is None:
            self.custom_cost_matrix_distance = SquaredDistance()
        else:
            self.custom_cost_matrix_distance = custom_cost_matrix_distance

        self.custom_bounding_matrix: np.ndarray = custom_bounding_matrix

    def _distance(self, x: np.ndarray, y: np.ndarray):
        """
        Method used to compute the distance between two timeseries

        Parameters
        ----------
        x: np.ndarray
            First time series
        y: np.ndarray
            Second time series

        Returns
        -------
        float
            Distance between time series x and time series y
        """

        bounding_matrix, pre_computed_distances = self._dtw_setup(x, y)

        # cost_matrix = _cost_matrix(x, y, bounding_matrix, pre_computed_distances)
        cost_matrix = _cost_matrix(x, y, bounding_matrix, pre_computed_distances)

        return np.sqrt(cost_matrix[-1, -1])

    def _check_params(self):
        """
        Method used to check the parameters of dtw

        Returns
        -------
        lower_bounding: LowerBounding
            The resolved LowerBounding object
        sakoe_chiba_window_radius: int
            Validated sakoe chiba window radius
        itakura_max_slope: float
            Validated itakura max slope radius
        """
        if isinstance(self.lower_bounding, int):
            lower_bounding = LowerBounding(self.lower_bounding)
        else:
            lower_bounding = LowerBounding.NO_BOUNDING

        (
            sakoe_chiba_window_radius,
            itakura_max_slope,
        ) = LowerBounding.check_bounding_parameters(
            self.sakoe_chiba_window_radius, self.itakura_max_slope
        )

        return lower_bounding, sakoe_chiba_window_radius, itakura_max_slope

    def _resolve_bounding_matrix(self, x, y):
        (
            lower_bounding,
            sakoe_chiba_window_radius,
            itakura_max_slope,
        ) = self._check_params()

        if self.custom_bounding_matrix is None:
            bounding_matrix = lower_bounding.create_bounding_matrix(
                x,
                y,
                sakoe_chiba_window_radius=sakoe_chiba_window_radius,
                itakura_max_slope=itakura_max_slope,
            )
        else:
            bounding_matrix = self.custom_bounding_matrix
        return bounding_matrix

    def _dtw_setup(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method used to validate and setup parameters to perform dtw

        Parameters
        ----------
        x: np.ndarray
            First time series
        y: np.ndarray
            Second time series

        Returns
        -------
        bounding_matrix: np.ndarray
            Bounding matrix for dtw
        pre_computed_distances: np.ndarray
            Pairwise matrix of distances between each point
        """
        bounding_matrix = self._resolve_bounding_matrix(x, y)

        if isinstance(self.custom_cost_matrix_distance, BaseDistance):
            pre_computed_distances = self.custom_cost_matrix_distance.pairwise(x, y)
        else:
            pre_computed_distances = self.custom_cost_matrix_distance(x, y)

        return bounding_matrix, pre_computed_distances

    def _numba_parameter_check(self, x, y):
        """
        Method used to check the incoming parameters for the numba distance function

        Parameters
        ----------
        x: np.ndarray
            First time series
        y: np.ndarray
            Second time series

        Returns
        -------
        np.ndarray
            Bounding matrix for dtw
        Callable
            Distance function to use in dtw
        """
        bounding_matrix = self._resolve_bounding_matrix(x, y)

        if isinstance(self.custom_cost_matrix_distance, NumbaSupportedDistance):
            dist_func = self.custom_cost_matrix_distance.numba_distance(x, y)
        elif isinstance(self.custom_cost_matrix_distance, Callable):
            dist_func = self.custom_cost_matrix_distance
        else:
            warnings.warn(
                "The current distance metric passed isn't numba compatible,"
                "defaulting to SquaredDistance"
            )
            dist_func = SquaredDistance().numba_distance(x, y)

        return bounding_matrix, dist_func

    def numba_distance(self, x, y) -> Callable[[np.ndarray, np.ndarray], float]:
        """
        Method used to return a numba callable distance, this assume that all checks
        have been done so the function returned doesn't need to check but checks
        should be done before the return

        Parameters
        ----------
        x: np.ndarray
            First time series
        y: np.ndarray
            Second time series

        Returns
        -------
        Callable
            Numba compiled function (i.e. has @njit decorator)
        """
        bounding_matrix, dist_func = self._numba_parameter_check(x, y)

        @njit()
        def numba_dtw(
            x: np.ndarray,
            y: np.ndarray,
            bounding_matrix: np.ndarray = bounding_matrix,
            dist_func: Callable = dist_func,
        ) -> float:
            symmetric = np.array_equal(x, y)

            computed_distances = _numba_pairwise(x, y, symmetric, dist_func)

            cost_matrix = _cost_matrix(x, y, bounding_matrix, computed_distances)

            return np.sqrt(cost_matrix[-1, -1])

        return numba_dtw
