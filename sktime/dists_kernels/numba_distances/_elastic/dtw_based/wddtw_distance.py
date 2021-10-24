# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

from typing import Union, Callable
import numpy as np

from sktime.dists_kernels.numba_distances._elastic.dtw_based.lower_bounding import (
    LowerBounding,
)
from sktime.dists_kernels.numba_distances._elastic.squared_distance import (
    _numba_squared_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw_based.wdtw_distance import (
    wdtw_distance,
    numba_wdtw_distance_factory,
)
from sktime.dists_kernels.numba_distances.pairwise_distances import pairwise_distance
from sktime.dists_kernels.numba_distances._elastic.dtw_based.ddtw_distance import (
    _pairwise_ddtw_param_validator,
    _diff_timeseries,
)


def wddtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
    bounding_matrix: np.ndarray = None,
    g: float = 0.05,
) -> float:
    """Method to calculate wddtw distance between timeseries.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        lower bounding technique to use. Potential bounding techniques and their int
        value are given below:
        NO_BOUNDING = 2
        SAKOE_CHIBA = 2
        ITAKURA_PARALLELOGRAM = 3
    window: int, defaults = 2
        Size of the bounding window
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura
    defaults = squared_distance
        Distance function to use
    distance: Callable[[np.ndarray, np.ndarray], float],
        defaults = squared_distance
        Distance function to use within dtw_based. Defaults to squared distance.
    bounding_matrix: np.ndarray, defaults = none
        Custom bounding matrix where inside bounding marked by finite values and
        outside marked with infinite values.
    g: float, defaults = 0.05
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase difference.
        tldr: it controls how aggressive the weighting reward or penalision is towards
        points.

    Returns
    -------
    float
        wddtw distance between the two timeseries
    """
    _x = _diff_timeseries(x)
    _y = _diff_timeseries(y)

    return wdtw_distance(
        _x,
        _y,
        lower_bounding=lower_bounding,
        window=window,
        itakura_max_slope=itakura_max_slope,
        distance=distance,
        bounding_matrix=bounding_matrix,
        g=g,
    )


def pairwise_wddtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
    bounding_matrix: np.ndarray = None,
    g: float = 0.05,
) -> np.ndarray:
    """Wddtw pairwise distance between two timeseries.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        lower bounding technique to use. Potential bounding techniques and their int
        value are given below:
        NO_BOUNDING = 2
        SAKOE_CHIBA = 2
        ITAKURA_PARALLELOGRAM = 3
    window: int, defaults = 2
        Size of the bounding window
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura
    defaults = squared_distance
        Distance function to use
    distance: Callable[[np.ndarray, np.ndarray], float],
        defaults = squared_distance
        Distance function to use within dtw_based. Defaults to squared distance.
    bounding_matrix: np.ndarray, defaults = none
        Custom bounding matrix where inside bounding marked by finite values and
        outside marked with infinite values.
    g: float, defaults = 0.05
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase difference.
        tldr: it controls how aggressive the weighting reward or penalision is towards
        points.

    Returns
    -------
    np.ndarray
        Pairwise matrix calculated using wddtw
    """
    return pairwise_distance(
        x,
        y,
        param_validator=_pairwise_ddtw_param_validator,
        numba_distance_factory=numba_wdtw_distance_factory,
        lower_bounding=lower_bounding,
        window=window,
        itakura_max_slope=itakura_max_slope,
        distance=distance,
        bounding_matrix=bounding_matrix,
        g=g,
    )
