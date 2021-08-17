# -*- coding: utf-8 -*-
__all__ = ["dtw", "dtw_and_cost_matrix"]

import numpy as np
from typing import Union
from enum import Enum
import math
from numba import njit

from sktime.metrics.distances._distance_utils import (
    format_distance_series,
)


class LowerBounding(Enum):
    """
    Enum for dtw lower bounding implementations

    Parameters
    ----------
    int_val: int
        integer value of the bounding to perform where:
        NO_BOUNDING = 1
        SAKOE_CHIBA = 2
        ITAKURA_PARALLELOGRAM = 3
    """

    NO_BOUNDING = 1  # No bounding on the matrix
    SAKOE_CHIBA = 2  # Sakoe Chiba bounding on the matrix
    ITAKURA_PARALLELOGRAM = 3  # Itakuras Parallelogram bounding on the matrix

    def __init__(self, int_val):
        self.int_val: int = int_val
        self.string_name: str = ""

    def create_bounding_matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sakoe_chiba_window_radius: int = 2,
        itakura_max_slope: float = 2.0,
    ) -> np.ndarray:
        """
        Method used to create the bounding matrix based on the enum

        Parameters
        ----------
        x: np.ndarray
            numpy array of first time series

        y: np.ndarray
            numpy array of second time series

        sakoe_chiba_window_radius: int, defaults = 2
            Integer that is the radius of the sakoe chiba window

        itakura_max_slope: float, defaults = 2.
            Gradient of the slope fo itakura

        Returns
        -------
        bounding_matrix: np.ndarray
            matrix where the cells inside the bound are 0s and anything outside the
            bound are infinity.
        """
        if self.int_val == 2:
            bounding_matrix = self.sakoe_chiba(x, y, sakoe_chiba_window_radius)
        elif self.int_val == 3:
            bounding_matrix = self.itakura_parallelogram(x, y, itakura_max_slope)
        else:
            bounding_matrix = self.no_bounding(x, y)

        return bounding_matrix

    def no_bounding(self, x: np.ndarray, y: np.ndarray):
        """
        Method used to get a matrix with no bounding

        Parameters
        ----------
        x: np.ndarray
            x array to calculate window for

        y: np.ndarray
            y array to calculate window for

        Returns
        -------
        np.ndarray
            bounding matrix with all values set to 0. as there is no bounding.
        """
        self._check_params(x, y)
        return np.zeros((x.shape[0], y.shape[0]))

    def sakoe_chiba(
        self, x: np.ndarray, y: np.ndarray, sakoe_chiba_window_radius: int
    ) -> np.ndarray:
        """
        Method used to calculate the sakoe chiba lower bounding window on a matrix

        Parameters
        ----------
        x: np.ndarray
            x array to calculate window for

        y: np.ndarray
            y array to calculate window for

        sakoe_chiba_window_radius: int
            integer that is the radius of the window

        Returns
        -------
        np.ndarray
            bounding matrix with the values inside the bound set to 0. and anything
            outside set to infinity
        """
        self._check_params(x, y)
        if not isinstance(sakoe_chiba_window_radius, int):
            raise ValueError("The sakoe chiba radius must be an integer.")

        bounding_matrix = np.full((y.shape[0], x.shape[0]), np.inf)

        x_size = x.shape[0]
        y_size = y.shape[0]

        x_upper_line_values = np.interp(
            list(range(x_size)),
            [0, x_size - 1],
            [0 - sakoe_chiba_window_radius, y_size - sakoe_chiba_window_radius - 1],
        )
        x_lower_line_values = np.interp(
            list(range(x_size)),
            [0, x_size - 1],
            [0 + sakoe_chiba_window_radius, y_size + sakoe_chiba_window_radius - 1],
        )

        bounding_matrix = self.create_shape_on_matrix(
            bounding_matrix, x_upper_line_values, x_lower_line_values
        )

        return bounding_matrix

    def itakura_parallelogram(
        self, x: np.ndarray, y: np.ndarray, itakura_max_slope: float
    ) -> np.ndarray:
        """
        Method used to calculate the itakura parallelogram bounding matrix

        Parameters
        ----------
        x: np.ndarray
            x array to calculate window for

        y: np.ndarray
            y array to calculate window for

        itakura_max_slope: float
            float that is the gradient of the slope

        Returns
        -------
        np.ndarray
            bounding matrix with the values inside the bound set to 0. and anything
            outside set to infinity

        """
        self._check_params(x, y)
        if not isinstance(itakura_max_slope, float):
            raise ValueError("The itakura max slope must be a float")

        bounding_matrix = np.full((y.shape[0], x.shape[0]), np.inf)

        x_size = x.shape[0]
        y_size = y.shape[0]

        middle_x_upper = math.ceil(x_size / 2)
        middle_x_lower = math.floor(x_size / 2)
        if middle_x_lower == middle_x_upper:
            middle_x_lower = middle_x_lower - 1
        middle_y = math.floor(y_size / 2)

        difference_from_middle_y = abs((middle_x_lower * itakura_max_slope) - middle_y)
        middle_y_lower = middle_y + difference_from_middle_y
        middle_y_upper = middle_y - difference_from_middle_y

        x_upper_line_values = np.interp(
            list(range(x_size)),
            [0, middle_x_lower, middle_x_upper, x_size - 1],
            [0, middle_y_upper, middle_y_upper, y_size - 1],
        )
        x_lower_line_values = np.interp(
            list(range(x_size)),
            [0, middle_x_lower, middle_x_upper, x_size - 1],
            [0, middle_y_lower, middle_y_lower, y_size - 1],
        )

        bounding_matrix = self.create_shape_on_matrix(
            bounding_matrix, x_upper_line_values, x_lower_line_values
        )

        return bounding_matrix

    @staticmethod
    @njit
    def create_shape_on_matrix(
        bounding_matrix: np.ndarray,
        y_upper_line: np.ndarray,
        y_lower_line: Union[np.ndarray, None] = None,
        x_step_size: int = 1,
        start_val: int = 0,
    ) -> np.ndarray:
        """
        Method used to create a shape from a given upper line and lower line on a matrix

        Parameters
        ----------
        bounding_matrix: np.ndarray
            matrix to copy, replace values in the shape and then return the copy

        y_upper_line: np.ndarray
            y points of the upper line

        y_lower_line: np.ndarray, defaults = None
            y points of the lower line. If no lower line specified, then y_upper_line
            used as lower line

        x_step_size: int, defaults = 1
            int that is the step size each iteration will increase by

        start_val: int, defaults = 0
            int that is the starting coordinate for x

        Returns
        -------
        np.ndarray
            Matrix with values of the shape set to 0.
        """
        y_size = bounding_matrix.shape[0]

        if y_lower_line is None:
            y_lower_line = y_upper_line

        upper_line_y_values = y_upper_line.shape[0]
        lower_line_y_values = y_lower_line.shape[0]

        if upper_line_y_values != lower_line_y_values:
            raise ValueError(
                "The number of upper line values must equal the number of lower line "
                "values"
            )

        half_way = math.floor(upper_line_y_values / 2)

        for i in range(start_val, upper_line_y_values):
            x = i * x_step_size

            if i > half_way:
                upper_y = max(0, min(y_size - 1, math.ceil(y_upper_line[i])))
                lower_y = max(0, min(y_size - 1, math.ceil(y_lower_line[i])))
            else:
                upper_y = max(0, min(y_size - 1, math.floor(y_upper_line[i])))
                lower_y = max(0, min(y_size - 1, math.floor(y_lower_line[i])))

            if upper_line_y_values == lower_line_y_values:
                if upper_y == lower_y:
                    bounding_matrix[upper_y, x] = 0.0
                else:
                    bounding_matrix[upper_y : lower_y + 1, x] = 0.0
            else:
                bounding_matrix[upper_y, x] = 0.0
                bounding_matrix[lower_y, x] = 0.0

        return bounding_matrix

    @staticmethod
    def _check_params(x: np.ndarray, y: np.ndarray):
        if x is None or y is None:
            raise ValueError("Both x and y values must be given.")


@njit
def _squared_dist(x, y):
    distance = 0.0
    for i in range(x.shape[0]):
        curr = x[i] - y[i]
        distance += curr * curr
    return distance


@njit
def _cost_matrix(x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray):
    """
    Method used to calculate the cost matrix to derive distance from

    Parameters
    ----------
    x: np.ndarray
        first time series

    y: np.ndarray
        second time series

    bounding_matrix: np.ndarray
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        curr_x = x[i]
        for j in range(y_size):
            if np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i + 1, j + 1] = _squared_dist(curr_x, y[j]) + min(
                    cost_matrix[i, j + 1], cost_matrix[i + 1, j], cost_matrix[i, j]
                )

    return cost_matrix[1:, 1:]


def _dtw_check_params(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
):
    x, y = format_distance_series(x, y)

    # TODO: check lengths of time series and interpolate on missing values

    if isinstance(lower_bounding, int):
        lower_bounding = LowerBounding(lower_bounding)

    return x, y, lower_bounding


def dtw(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    sakoe_chiba_window_radius: int = 2,
    itakura_max_slope: float = 2.0,
):
    """
    Method used to calculate the dynamic time warping distance between two time series

    x: np.ndarray
        first time series

    y: np.ndarray
        second time series

    lower_bounding: LowerBounding or int, defaults = NO_BOUNDING
        Lower bounding algorithm to use. The following describes the potential
        parameters:
        no bounding if LowerBounding.NO_BOUNDING or 1
        sakoe chiba bounding if LowerBounding.SAKOE_CHIBA or 2
        itakura parallelogram if LowerBounding.ITAKURA_PARALLELOGRAM or 3


    sakoe_chiba_window_radius: int, defaults = 2
        Integer that is the radius of the sakoe chiba window

    itakura_max_slope: float, defaults = 2.
        Gradient of the slope fo itakura

    Returns
    -------
        float that is the distance between the two time series
    """

    # sakoe_chiba_window_radius and itakura_max_slope are checked when LowerBounding
    # enum created so dont need to check here
    x, y, lower_bounding = _dtw_check_params(x, y, lower_bounding)

    bounding_matrix = lower_bounding.create_bounding_matrix(
        x, y, sakoe_chiba_window_radius, itakura_max_slope
    )
    cost_matrix = _cost_matrix(x, y, bounding_matrix)
    return np.sqrt(cost_matrix[-1, -1])


def dtw_and_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    sakoe_chiba_window_radius: int = 2,
    itakura_max_slope: float = 2.0,
):
    """
    Method used to calculate the dynamic time warping between two series, returning
    both the dtw between x and y and the cost matrix used to generate the distance.

    x: np.ndarray
        first time series

    y: np.ndarray
        second time series

    lower_bounding: LowerBounding or int, defaults = NO_BOUNDING
        Lower bounding algorithm to use. The following describes the potential
        parameters:
        no bounding if LowerBounding.NO_BOUNDING or 1
        sakoe chiba bounding if LowerBounding.SAKOE_CHIBA or 2
        itakura parallelogram if LowerBounding.ITAKURA_PARALLELOGRAM or 3


    sakoe_chiba_window_radius: int, defaults = 2
        Integer that is the radius of the sakoe chiba window

    itakura_max_slope: float, defaults = 2.
        Gradient of the slope fo itakura

    Returns
    -------
    distance: float
        dtw distance between x and y
    cost_matrix: np.ndarray
        cost matrix used to generate the dtw between the two series
    """

    # sakoe_chiba_window_radius and itakura_max_slope are checked when LowerBounding
    # enum created so dont need to check here
    x, y, lower_bounding = _dtw_check_params(x, y, lower_bounding)

    bounding_matrix = lower_bounding.create_bounding_matrix(
        x, y, sakoe_chiba_window_radius, itakura_max_slope
    )

    cost_matrix = _cost_matrix(x, y, bounding_matrix)
    return np.sqrt(cost_matrix[-1, -1]), cost_matrix


# def dtw_pairwise(
#         x: np.ndarray,
#         y: np.ndarray = None,
#         lower_bounding=None,
#         sakoe_chiba_window_radius=None,
#         itakura_max_slope=None,
# ):
#     kwargs = {
#         "lower_bounding": lower_bounding,
#         "sakoe_chiba_window_radius": sakoe_chiba_window_radius,
#         "itakura_max_slopes": itakura_max_slope,
#     }
#     return _DtwDistance().pairwise(x, y, **kwargs)
#
#
# @dataclass(frozen=True)
# class DistanceInfo:
#     """
#     Dataclass used to register valid distance metrics. This contains all the
#     info you need for cdists and pdists and additional info such as str values
#     for the distance metric
#     """
#
#     # Name of python distance function
#     canonical_name: str
#     # All aliases, including canonical_name
#     aka: Set[str]
#     # Base distance class
#     distance_class: Optional[Type[BaseDistance]]
#
#
# # Registry of implemented metrics:
# DISTANCE_INFO = [
#     DistanceInfo(
#         canonical_name="dtw",
#         aka={"dtw", "dynamic time warping"},
#         distance_class=_DtwDistance,
#     ),
#     DistanceInfo(
#         canonical_name="dtw cost matrix",
#         aka={"dtw cost matrix", "dynamic time warping cost matrix"},
#         distance_class=_DtwDistanceCostMatrix,
#     ),
# ]
#
#
# @dataclass(frozen=True)
# class CDistWrapper:
#     metric_name: str
#
#     def __call__(self, x, y, **kwargs):
#         pass
#
#
# @dataclass(frozen=True)
# class PDistWrapper:
#     metric_name: str
#
#     def __call__(self, x, y, **kwargs):
#         pass
#
#
# _scipy_distances = [
#     "braycurtis",
#     "canberra",
#     "chebyshev",
#     "cityblock",
#     "correlation",
#     "cosine",
#     "dice",
#     "euclidean",
#     "hamming",
#     "jaccard",
#     "jensenshannon",
#     "kulsinski",
#     "mahalanobis",
#     "matching",
#     "minkowski",
#     "rogerstanimoto",
#     "russellrao",
#     "seuclidean",
#     "sokalmichener",
#     "sokalsneath",
#     "sqeuclidean",
#     "yule",
# ]
