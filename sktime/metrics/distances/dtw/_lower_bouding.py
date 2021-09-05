# -*- coding: utf-8 -*-
import numpy as np
from typing import Union, List
from enum import Enum
import math
from numba import njit


@njit(cache=True)
def _plot_values_on_matrix(bounding_matrix: np.ndarray, values: np.ndarray):
    for i in range(values.shape[0]):
        arr = values[i]
        bounding_matrix[arr[0], arr[1]] = 0.0
    return bounding_matrix


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
            Gradient of the slope of itakura

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

        if np.array_equal(
            np.array((x_upper_line_values)), np.array(x_lower_line_values)
        ):
            x_upper_line_values = self._check_line_steps(x_upper_line_values)

        bounding_matrix = self.create_shape_on_matrix(
            bounding_matrix, x_upper_line_values, x_lower_line_values
        )

        return bounding_matrix

    @staticmethod
    def _check_line_steps(line: Union[List, np.ndarray]):
        """
        Method that ensures that the line steps are within one of each other or dtw
        wont work

        Parameters
        ----------
        line: List or np.ndarray
            line to check steps

        Returns
        -------
        List or np.ndarray
            Line with updated steps if they were out of line
        """
        prev = line[0]
        for i in range(1, len(line)):
            curr_val = line[i]
            if curr_val > (prev + 1):
                line[i] = prev + 1
            elif curr_val < (prev - 1):
                line[i] = prev - 1
            prev = curr_val
        return line

    @staticmethod
    @njit(cache=True)
    def plot_values_on_matrix(bounding_matrix: np.ndarray, values: np.ndarray):
        return _plot_values_on_matrix(bounding_matrix, values)

    @staticmethod
    @njit(cache=True)
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
    def check_bounding_parameters(
        sakoe_chiba_window_radius: int = 2, itakura_max_slope: float = 2.0
    ):
        """

        Parameters
        ----------
        sakoe_chiba_window_radius: int, defaults = 2
            Integer that is the radius of the sakoe chiba window

        itakura_max_slope: float, defaults = 2.
            Gradient of the slope of itakura
        Returns
        -------
        sakoe_chiba_window_radius: int
            Integer that is the radius of the sakoe chiba window
        itakura_max_slope: float
            Gradient of the slope of itakura
        """
        if not isinstance(sakoe_chiba_window_radius, int):
            raise ValueError("The sakoe chiba radius must be an int value")
        if not isinstance(itakura_max_slope, float):
            raise ValueError("The itakura max slope must be a float value")
        return sakoe_chiba_window_radius, itakura_max_slope
