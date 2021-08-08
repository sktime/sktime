# -*- coding: utf-8 -*-
import math
from enum import Enum
import numpy as np
from typing import Union
from numba import prange, njit

from sktime.dists_kernels.distances.base.base import BaseDistance

np_or_none = Union[np.ndarray, None]


class LowerBounding(Enum):
    NO_BOUNDING = (1, "no bounding")
    SAKOE_CHIBA = (2, "sakoe chiba")
    ITAKURA_PARALLELOGRAM = (3, "itakura parallelogram")

    def __init__(self, int_val, string_name):
        self.int_val: int = int_val
        self.string_name: str = string_name

    def create_bounding_matrix(
        self, x: np.ndarray, y: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Method used to create the bounding matrix based on the enum

        Parameters
        ----------
        x: np.ndarray
            numpy array of first time series

        y: np.ndarray
            numpy array of second time series

        Returns
        -------
        bounding_matrix: np.ndarray
            matrix where the cells inside the bound are 0s and anything outside the
            bound are infinity.
        """
        if self.int_val == 2:
            bounding_matrix = self.sakoe_chiba(
                x, y, kwargs.get("sakoe_chiba_window_radius")
            )
        elif self.int_val == 3:
            bounding_matrix = self.itakura_parallelogram(
                x, y, kwargs.get("itakura_max_slope")
            )
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
            [0, x_size],
            [0 - sakoe_chiba_window_radius, y_size - sakoe_chiba_window_radius],
        )
        x_lower_line_values = np.interp(
            list(range(x_size)),
            [0, x_size],
            [0 + sakoe_chiba_window_radius, y_size + sakoe_chiba_window_radius],
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
            [0, middle_x_lower, middle_x_upper, x_size],
            [0, middle_y_upper, middle_y_upper, y_size],
        )
        x_lower_line_values = np.interp(
            list(range(x_size)),
            [0, middle_x_lower, middle_x_upper, x_size],
            [0, middle_y_lower, middle_y_lower, y_size],
        )

        bounding_matrix = self.create_shape_on_matrix(
            bounding_matrix, x_upper_line_values, x_lower_line_values
        )

        return bounding_matrix

    @staticmethod
    def create_shape_on_matrix(
        bounding_matrix: np.ndarray,
        y_upper_line: np.ndarray,
        y_lower_line: np_or_none = None,
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

        for i in prange(start_val, upper_line_y_values):
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


# dtw based distances
class _DtwDistance(BaseDistance):
    def __init__(self):
        pass

    def _distance(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Method used to calculate the dtw distance between two time series

        Parameters
        ----------
        x: np.ndarray
            time series to find distance from

        y: np.ndarray
            time series to find distance from
        """
        return _DtwDistance._dtw_distance(x, y)

    @staticmethod
    @njit(nopython=True)
    def _dtw_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sum(np.square(x - y))

    @staticmethod
    @njit(nopython=True)
    def _cost_matrix(x, y, mask):
        pass


dtw = _DtwDistance()
