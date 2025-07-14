"""Lower bounding enum."""

__author__ = ["chrisholder", "TonyBagnall"]
__all__ = ["LowerBounding", "resolve_bounding_matrix"]

from enum import Enum
from typing import Union

import numpy as np


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
    .. [1]  H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for
            spoken word recognition," IEEE Transactions on Acoustics, Speech and
            Signal Processing, vol. 26, no. 1, pp. 43--49, 1978.
    .. [2]  F. Itakura, "Minimum prediction residual principle applied to speech
            recognition," IEEE Transactions on Acoustics, Speech, and Signal
            Processing, vol. 23, no. 1, pp. 67-72, February 1975,
            doi: 10.1109/TASSP.1975.1162641.
    .. [3]  C. A. Ratanamahatana and E. Keogh, "Making Time-Series
            Classification More Accurate Using Learned Constraints," in Proceedings
            of the 2004 SIAM International Conference on Data Mining (SDM), 2004,
            doi: 10.1137/1.9781611972740.2.
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
        sakoe_chiba_window_radius: Union[float, None] = None,
        itakura_max_slope: Union[float, int, None] = None,
    ) -> np.ndarray:
        """Create a bounding matrix.

        The bounding matrix that will be created is the one the enum is initialised
        as.

        Parameters
        ----------
        x: np.ndarray (1d, 2d or 3d array)
            First time series.
        y: np.ndarray (1d, 2d or 3d array)
            Second time series.
        sakoe_chiba_window_radius: int, defaults = None
            Integer that is the radius of the sakoe chiba window. Must be between 0
            and 1.
        itakura_max_slope: float or int, defaults = None
            Gradient of the slope for itakura parallelogram. Must be between 0 and 1.

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
            If the input time series is not a numpy array.
            If the input time series doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
        """
        from sktime.distances._lower_bounding_numba import (
            itakura_parallelogram,
            no_bounding,
            sakoe_chiba,
        )

        _x = self._check_input_timeseries(x)
        _y = self._check_input_timeseries(y)
        if self.int_val == 2:
            if not isinstance(sakoe_chiba_window_radius, float):
                raise ValueError(
                    f"The sakoe chiba window must be a float, passed "
                    f"{sakoe_chiba_window_radius} of type"
                    f" {type(sakoe_chiba_window_radius)}."
                )
            bounding_matrix = sakoe_chiba(_x, _y, sakoe_chiba_window_radius)
        elif self.int_val == 3:
            if not isinstance(itakura_max_slope, float):
                raise ValueError("The itakura max slope must be a float or int.")
            bounding_matrix = itakura_parallelogram(_x, _y, itakura_max_slope)
        else:
            bounding_matrix = no_bounding(_x, _y)

        return bounding_matrix

    @staticmethod
    def _check_input_timeseries(x: np.ndarray) -> np.ndarray:
        """Check and validate input time series.

        Parameters
        ----------
        x: np.ndarray (1d, 2d or 3d array)
            A time series.

        Returns
        -------
        np.ndarray (2d array)
            A validated time series.

        Raises
        ------
        ValueError
            If the input time series is not a numpy array.
            If the input timen series doesn't have exactly 2 dimensions.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("The input time series must be a numpy array.")
        if x.ndim <= 0 or x.ndim >= 4:
            raise ValueError(
                "The input time series must have more than 0 dimensions and"
                "less than 4 dimensions."
            )
        if x.ndim == 3:
            return x[0]
        return x


def resolve_bounding_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Union[float, None] = None,
    itakura_max_slope: Union[float, None] = None,
    bounding_matrix: np.ndarray = None,
) -> np.ndarray:
    """Resolve the bounding matrix parameters.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series.
    y: np.ndarray (2d array)
        Second time series.
    window: float, defaults = None
        Float that is the % radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding). Must be between 0 and 1.
    itakura_max_slope: float, defaults = None
        Gradient of the slope for itakura parallelogram (if using Itakura
        Parallelogram lower bounding). Must be between 0 and 1.
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
        If the input time series is not a numpy array.
        If the input time series doesn't have exactly 2 dimensions.
        If the sakoe_chiba_window_radius is not an float.
        If the itakura_max_slope is not a float or int.
        If both window and itakura_max_slope are set
    """
    if bounding_matrix is None:
        if itakura_max_slope is not None and window is not None:
            raise ValueError(
                "You can only use one bounding matrix at once. You"
                "have set both window and itakura_max_slope parameter."
            )
        if window is not None:
            # Sakoe-chiba
            lower_bounding = LowerBounding(2)
        elif itakura_max_slope is not None:
            # Itakura parallelogram
            lower_bounding = LowerBounding(3)
        else:
            # No bounding
            lower_bounding = LowerBounding(1)

        return lower_bounding.create_bounding_matrix(
            x,
            y,
            sakoe_chiba_window_radius=window,
            itakura_max_slope=itakura_max_slope,
        )
    else:
        return bounding_matrix
