# -*- coding: utf-8 -*-
"""Lower bounding enum."""

__author__ = ["chrisholder"]
__all__ = ["LowerBounding", "resolve_bounding_matrix"]

import math
from enum import Enum
from typing import Union

import numpy as np
from numba import njit


class LowerBounding(Enum):
    r"""Enum for various lower bounding implementations.

    Bounding techniques constrain the indices a warping path may take. It does this
    by defining a criteria that every index in a warping path must conform to.
    We can define this bounding as:

    .. math::
        j-R_{i} \leq i \leq j+R_{i}

    Where i is the ith index value, j is the jth index value and R is a term defining
    the range in which a index is in bound.  It should be noted can be independent of
    the indexes (such as in Sakoe-Chiba bounding) or can even be a function of the index
    (such as in Itakura Parallelogram).

    The available bounding techniques are defined below:

        'NO_BOUNDING'. A warping path using no bounding is said to have no
        restrictions in what indexes are considered for warping. For no bounding R
        can be defined as:

        .. math::
            R = max(len(x), len(y))

        'SAKOE_CHIBA'. A warping path using Sakoe-Chibas technique restricts the warping
        path using a constant defined window size. This window determines the
        largest temporal shift allowed from the diagonal. This was originally presented
        in [1]_. For Sakoe-Chiba R can be defined as:

        .. math::
            R = window_size

        'ITAKURA_PARALLELOGRAM'. A warping path using Itakura-Parallelogram restricts
        the warping path as a function of a defined max slope value.
        This creates a parallelogram shape on the warping path.
        For Itakura-Parallelogram R is therefore instead defined as a function.
        Itakura parallelogram was originally presented in [2]_.

    For a more general comparison between different bounding see [3]_.

    Parameters
    ----------
    int_val: int
        integer value to perform desired bounding, where:
        NO_BOUNDING = 1
        SAKOE_CHIBA = 2
        ITAKURA_PARALLELOGRAM = 3

    References
    ----------
    .. [1]  H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
            spoken word recognition," IEEE Transactions on Acoustics, Speech and
            Signal Processing, vol. 26(1), pp. 43--49, 1978.
    .. [2]  F. Itakura, "Minimum prediction residual principle applied to speech
            recognition," in IEEE Transactions on Acoustics, Speech, and Signal
            Processing, vol. 23, no. 1, pp. 67-72, February 1975,
            doi: 10.1109/TASSP.1975.1162641.
    .. [3]  Ratanamahatana, Chotirat & Keogh, Eamonn. (2004). Making Time-Series
            Classification More Accurate Using Learned Constraints.
            10.1137/1.9781611972740.2.

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
        itakura_max_slope: Union[float, int] = 2.0,
    ) -> np.ndarray:
        """Create a bounding matrix.

        The bounding matrix that will be created is the one the enum is initialised
        as.

        Parameters
        ----------
        x: np.ndarray (1d, 2d or 3d array)
            First timeseries.
        y: np.ndarray (1d, 2d or 3d array)
            Second timeseries.
        sakoe_chiba_window_radius: int, defaults = 2
            Integer that is the radius of the sakoe chiba window.
        itakura_max_slope: float or int, defaults = 2.
            Gradient of the slope for itakura parallelogram.

        Returns
        -------
        np.ndarray (2d of size mxn where m is len(x) and n is len(y). If 3d array passed
                    x then m is len(x[0]) and n is len(y[0]).
            Bounding matrix where the values inside the bound are finite values (0s) and
            outside the bounds are infinity (non finite). This allows you to
            check if a given index is in bound using np.isfinite(bounding_matrix[i, j]).

        Raises
        ------
        ValueError
            If the input timeseries is not a numpy array.
            If the input timeseries doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
        """
        _x = self._check_input_timeseries(x)
        _y = self._check_input_timeseries(y)
        if self.int_val == 2:
            bounding_matrix = self.sakoe_chiba(_x, _y, sakoe_chiba_window_radius)
        elif self.int_val == 3:
            bounding_matrix = self.itakura_parallelogram(_x, _y, itakura_max_slope)
        else:
            bounding_matrix = self.no_bounding(_x, _y)

        return bounding_matrix

    def no_bounding(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create a matrix with no bounding.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First timeseries.
        y: np.ndarray (2d array)
            Second timeseries.

        Returns
        -------
        np.ndarray (2d of size mxn where m is len(x) and n is len(y)).
            Bounding matrix where the values inside the bound are finite values (0s) and
            outside the bounds are infinity (non finite).
        """
        return np.zeros((x.shape[0], y.shape[0]))

    def sakoe_chiba(
        self, x: np.ndarray, y: np.ndarray, sakoe_chiba_window_radius: int
    ) -> np.ndarray:
        """Create a sakoe chiba lower bounding window on a matrix.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First timeseries.
        y: np.ndarray (2d array)
            Second timeseries.
        sakoe_chiba_window_radius: int
            Integer that is the radius of the window.

        Returns
        -------
        np.ndarray (2d of size mxn where m is len(x) and n is len(y)).
            Sakoe Chiba bounding matrix where the values inside the bound are finite
            values (0s) and outside the bounds are infinity (non finite).

        Raises
        ------
        ValueError
            If the sakoe_chiba_window_radius is not an integer.
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
        """Create a itakura parallelogram bounding matrix.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First timeseries.
        y: np.ndarray (2d array)
            Second timeseries.
        itakura_max_slope: float or int
            Gradient of the slope.

        Returns
        -------
        np.ndarray (2d of size mxn where m is len(x) and n is len(y)).
            Sakoe Chiba bounding matrix where the values inside the bound are finite
            values (0s) and outside the bounds are infinity (non finite).

        Raises
        ------
        ValueError
            If the itakura_max_slope is not a float or int.
        """
        if not isinstance(itakura_max_slope, float):
            if isinstance(itakura_max_slope, int):
                itakura_max_slope = float(itakura_max_slope)
            else:
                raise ValueError("The itakura max slope must be a float or int.")

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

        if np.array_equal(np.array(x_upper_line_values), np.array(x_lower_line_values)):
            x_upper_line_values = self._check_line_steps(x_upper_line_values)

        bounding_matrix = self.create_shape_on_matrix(
            bounding_matrix, x_upper_line_values, x_lower_line_values
        )

        return bounding_matrix

    @staticmethod
    def _check_input_timeseries(x: np.ndarray) -> np.ndarray:
        """Check and validate input timeseries.

        Parameters
        ----------
        x: np.ndarray (1d, 2d or 3d array)
            A timeseries.

        Returns
        -------
        np.ndarray (2d array)
            A validated timeseries.

        Raises
        ------
        ValueError
            If the input timeseries is not a numpy array.
            If the input timeseries doesn't have exactly 2 dimensions.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("The input timeseries must be a numpy array.")
        if x.ndim <= 0 or x.ndim >= 4:
            raise ValueError(
                "The input timeseries must have more than 0 dimensions and"
                "less than 4 dimensions."
            )
        if x.ndim == 3:
            return x[0]
        return x

    @staticmethod
    def _check_line_steps(line: np.ndarray) -> np.ndarray:
        """Check the next 'step' is along the line.

        Parameters
        ----------
        line: np.ndarray
            line to check steps

        Returns
        -------
        np.ndarray
            Line with updated indexes.
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
    @njit()
    def create_shape_on_matrix(
        bounding_matrix: np.ndarray,
        y_upper_line: np.ndarray,
        y_lower_line: Union[np.ndarray, None] = None,
        x_step_size: int = 1,
        start_val: int = 0,
    ) -> np.ndarray:
        """Create a shape from a given upper line and lower line on a matrix.

        Parameters
        ----------
        bounding_matrix: np.ndarray (2d array)
            Matrix of size mxn where m is len(x) and n is len(y). Values that
            are inside the shape will be replaced with finite values (0.).
        y_upper_line: np.ndarray (1d array)
            Y points of the upper line.
        y_lower_line: np.ndarray (1d array), defaults = None
            Y points of the lower line. If no lower line specified, then y_upper_line
            used as lower line.
        x_step_size: int, defaults = 1
            Step size each iteration will increase by
        start_val: int, defaults = 0
            Starting coordinate for x

        Returns
        -------
        np.ndarray (2d array)
            Matrix with values of the shape set to 0. (finite), of the same shape
            as the passed bounding_matrix.
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
                    bounding_matrix[upper_y : (lower_y + 1), x] = 0.0
            else:
                bounding_matrix[upper_y, x] = 0.0
                bounding_matrix[lower_y, x] = 0.0

        return bounding_matrix


def resolve_bounding_matrix(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    bounding_matrix: np.ndarray = None,
) -> np.ndarray:
    """Resolve the bounding matrix parameters.

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
    bounding_matrix: np.ndarray (2d array)
        Custom bounding matrix to use. If defined then this matrix will be returned.
        Other lower_bounding params and creation will be ignored. The matrix should be
        structure so that indexes considered in bound should be the value 0. and indexes
        outside the bounding matrix should be infinity.

    Returns
    -------
    np.ndarray (2d array)
        Bounding matrix to use. The matrix is structured so that indexes
        considered in bound are of the value 0. and indexes outside the bounding
        matrix of the value infinity.

    Raises
    ------
    ValueError
        If the input timeseries is not a numpy array.
        If the input timeseries doesn't have exactly 2 dimensions.
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
    """
    if bounding_matrix is None:
        if isinstance(lower_bounding, int):
            lower_bounding = LowerBounding(lower_bounding)
        else:
            lower_bounding = lower_bounding

        return lower_bounding.create_bounding_matrix(
            x,
            y,
            sakoe_chiba_window_radius=window,
            itakura_max_slope=itakura_max_slope,
        )
    else:
        return bounding_matrix
