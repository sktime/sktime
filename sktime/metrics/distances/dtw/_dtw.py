# -*- coding: utf-8 -*-
from typing import Union, Tuple, Callable
import numpy as np
from numba import njit

from sktime.metrics.distances.dtw._lower_bouding import LowerBounding
from sktime.metrics.distances.base.base import BaseDistance, BasePairwise
from sktime.metrics.distances._squared_dist import SquaredDistance


@njit
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


class Dtw(BaseDistance, BasePairwise):
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
        Distance pairwise to use a custom distance for calculating cost matrix. If
        passing a custom Callable it takes the form: func(x, y) -> np.ndarray
        where the return is a matrix of the pairwise distance
    """

    def __init__(
        self,
        lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
        sakoe_chiba_window_radius: int = 2,
        itakura_max_slope: float = 2.0,
        custom_cost_matrix_distance: Union[BasePairwise, Callable] = SquaredDistance(),
    ):
        super(Dtw, self).__init__("dtw", {"dynamic me warping"})
        self.lower_bounding: Union[LowerBounding, int] = lower_bounding
        self.sakoe_chiba_window_radius: int = sakoe_chiba_window_radius
        self.itakura_max_slope: float = itakura_max_slope
        self.custom_cost_matrix_distance: Union[
            BasePairwise, Callable
        ] = custom_cost_matrix_distance

    def _distance(self, x: np.ndarray, y: np.ndarray):
        """
        Method used to compute the distance between two ts series

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

        cost_matrix = _cost_matrix(x, y, bounding_matrix, pre_computed_distances)

        return np.sqrt(cost_matrix[-1, -1])

    def _pairwise(self, x: np.ndarray, y: np.ndarray, symmetric: bool) -> np.ndarray:
        """
        Method to compute a pairwise distance on a matrix (i.e. distance between each
        ts in the matrix)

        Parameters
        ----------
        x: np.ndarray
            First matrix of multiple time series
        y: np.ndarray
            Second matrix of multiple time series.
        symmetric: bool
            boolean that is true when the two time series are equal to each other

        Returns
        -------
        np.ndarray
            Matrix containing the pairwise distance between each point
        """

        bounding_matrix, pre_computed_distances = self._dtw_setup(x, y)

        @njit
        def pairwise_wrapper(x: np.ndarray, y: np.ndarray) -> float:
            cost_matrix = _cost_matrix(x, y, bounding_matrix, pre_computed_distances)
            return np.sqrt(cost_matrix[-1, -1])

        distance_matrix = self.compute_pairwise_matrix(
            x, y, symmetric, pairwise_wrapper
        )
        return distance_matrix

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
        (
            lower_bounding,
            sakoe_chiba_window_radius,
            itakura_max_slope,
        ) = self._check_params()

        bounding_matrix = lower_bounding.create_bounding_matrix(
            x, y, sakoe_chiba_window_radius, itakura_max_slope
        )

        if isinstance(self.custom_cost_matrix_distance, BasePairwise):
            pre_computed_distances = self.custom_cost_matrix_distance.pairwise(x, y)
        else:
            pre_computed_distances = self.custom_cost_matrix_distance(x, y)

        return bounding_matrix, pre_computed_distances
