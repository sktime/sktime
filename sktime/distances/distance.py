# -*- coding: utf-8 -*-
"""Compute the distance between two timeseries."""

from typing import Callable, Union

import numpy as np

from sktime.distances._ddtw import DerivativeCallable, _average_of_slope, _DdtwDistance
from sktime.distances._dtw import _DtwDistance
from sktime.distances._edr import _EdrDistance
from sktime.distances._erp_distance import _ErpDistance
from sktime.distances._euclidean import _EuclideanDistance
from sktime.distances._lcss import _LcssDistance
from sktime.distances._numba_utils import (
    _compute_distance,
    _compute_pairwise_distance,
    to_numba_pairwise_timeseries,
    to_numba_timeseries,
)
from sktime.distances._resolve_metric import _resolve_metric
from sktime.distances._squared import _SquaredDistance
from sktime.distances._wddtw import _WddtwDistance
from sktime.distances._wdtw import _WdtwDistance
from sktime.distances.base import DistanceCallable, MetricInfo, NumbaDistance
from sktime.distances.lower_bounding import LowerBounding


def erp_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    custom_distance: DistanceCallable = _EuclideanDistance().distance_factory,
    bounding_matrix: np.ndarray = None,
    g: float = 0.0,
    **kwargs: dict,
) -> float:
    """Compute the Edit distance for real penalty (erp) distance between two series.

    Erp first proposed in [1]_ aims to improve on both dtw. Erp attempts
    to improve accuracy of distance computation by considering how we carry forward
    values that are 'gaps'. Gaps are defined in dtw when we choose which element
    to carry forward. When computing the cost matrix we use:

    min(cost_matrix[i, j],  cost_matrix[i - 1, j], cost_matrix[i, j - 1]

    If cost_matrix[i, j] is chosen this means there is no gap and an alignment has
    been found to the next element. If either cost_matrix[i - 1, j] or
    cost_matrix[i, j - 1] are chosen this means there is a gap and no alignment
    can be found to the next element. It is important to note when there is a gap
    a value is assigned. In dtw this value is going to greatly vary in size because
    it is derived from the computation between two points. This means that there
    is an inconsistent penalty for gaps being found. Erp attempts to solve this by
    defining a constant 'g' that when a gap is found, the distance between 'g' and
    the unmatched point are taken to determine how detrimental the gap is to
    the distance computation. By using a constant to compare to this gives a consistent
    measure of detriment.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
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
    g: float, defaults = 0.
        The reference value to penalise gaps.
    kwargs: dict
        Extra arguments for custom distance should be put in the kwargs. See the
        documentation for the distance for kwargs.

    Returns
    -------
    float
        erp distance between two timeseries.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If the metric type cannot be determined

    References
    ----------
    .. [1] Lei Chen and Raymond Ng. 2004. On the marriage of Lp-norms and edit distance.
    In Proceedings of the Thirtieth international conference on Very large data bases
     - Volume 30 (VLDB '04). VLDB Endowment, 792–803.
    """
    format_kwargs = {
        "lower_bounding": lower_bounding,
        "window": window,
        "itakura_max_slope": itakura_max_slope,
        "custom_distance": custom_distance,
        "bounding_matrix": bounding_matrix,
        "g": g,
    }
    format_kwargs = {**format_kwargs, **kwargs}

    return distance(x, y, metric="erp", **format_kwargs)


def edr_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    custom_distance: DistanceCallable = _SquaredDistance().distance_factory,
    bounding_matrix: np.ndarray = None,
    epsilon: float = 1.0,
    **kwargs: dict,
) -> float:
    """Compute the Edit distance for real sequences (edr) distance between two series.

    Edr computes the minimum number of elements (as a percentage) that must be removed
    from x and y so that the sum of the distance between the remaining signal elements
    lies within the tolerance (epsilon). Edr was originally put forward in [1]_.

    The value returned will be between 0 and 1 per time series (if a panel (3d array)
    is passed the value will be between 0 and max(len(x), len(y)). The value will
    represent as a percentage of elements that must be removed for the timeseries to
    be an exact match.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
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
    epsilon : float, defaults = 1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'.
    kwargs: dict
        Extra arguments for custom distance should be put in the kwargs. See the
        documentation for the distance for kwargs.

    Returns
    -------
    float
        edr distance between the two timeseries. The value will be between 0.0 and 1.0
        (unless panel is passed the value will be between 0 and max(len(x), len(y)),
        where 0.0 is an exact match between timeseries (i.e. they are the same) and
        1.0 where the are no matching subsequences.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If the metric type cannot be determined

    References
    ----------
    .. [1] Lei Chen, M. Tamer Özsu, and Vincent Oria. 2005. Robust and fast similarity
    search for moving object trajectories. In Proceedings of the 2005 ACM SIGMOD
    international conference on Management of data (SIGMOD '05). Association for
    Computing Machinery, New York, NY, USA, 491–502.
    DOI:https://doi.org/10.1145/1066157.1066213
    """
    format_kwargs = {
        "lower_bounding": lower_bounding,
        "window": window,
        "itakura_max_slope": itakura_max_slope,
        "custom_distance": custom_distance,
        "bounding_matrix": bounding_matrix,
        "epsilon": epsilon,
    }
    format_kwargs = {**format_kwargs, **kwargs}

    return distance(x, y, metric="edr", **format_kwargs)


def lcss_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    custom_distance: DistanceCallable = _SquaredDistance().distance_factory,
    bounding_matrix: np.ndarray = None,
    epsilon: float = 1.0,
    **kwargs: dict,
) -> float:
    """Compute the longest common subsequence (lcss) score between two timeseries.

    Lcss attempts to find the longest common sequence between two timeseries and returns
    a value that is the percentage that longest common sequence assumes. Originally
    present in [1]_, lcss is computed by matching indexes that are similar up until a
    defined threshold (epsilon). Matches can occur even if the time indexes are
    different. How far the time index difference can be is controlled by the
    the bounding matrix used.

    The value returned will be between 0 and 1 (0 is 100% subsequence match, 1 is 0%
    subsequence match) per time series (this means if a panel
    is passed the value will be between 0 and max(len(x), len(y)). The value will
    represent will represent a percentage over the timeseries of the longest common
    sequence occurred.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
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
    epsilon : float, defaults = 1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'.
    kwargs: dict
        Extra arguments for custom distance should be put in the kwargs. See the
        documentation for the distance for kwargs.

    Returns
    -------
    float
        lcss score between the two timeseries. The value will be between 0.0 and 1.0
        (unless panel is passed the value will be between 0 and max(len(x), len(y)),
        where 1.0 is an exact match between timeseries (i.e. they are the same) and
        0.0 where the are no matching subsequences.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If the metric type cannot be determined

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
        Similar Multidimensional Trajectories", In Proceedings of the
        18th International Conference on Data Engineering (ICDE '02).
        IEEE Computer Society, USA, 673.
    """
    format_kwargs = {
        "lower_bounding": lower_bounding,
        "window": window,
        "itakura_max_slope": itakura_max_slope,
        "custom_distance": custom_distance,
        "bounding_matrix": bounding_matrix,
        "epsilon": epsilon,
    }
    format_kwargs = {**format_kwargs, **kwargs}

    return distance(x, y, metric="lcss", **format_kwargs)


def wddtw_distance(
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
) -> float:
    r"""Compute the weighted derivative dynamic time warping (Wddtw) distance.

    Wddtw was first proposed in [1]_ as a further extension to ddtw. By adding a weight
    to the derivative it means the alignment isn't only considering the shape of the
    timeseries (gained from taking the derivative), but also the phase.

    Formally the derivative is calculated as:

    .. math::
        D_{x}[q] = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Which a weighted derivative can be calculated using D (the derivative) as:

    .. math::
        d_{w}(x_{i}, y_{j}) = ||w_{|i-j|}(D_{x_{i}} - D_{y_{j}})||

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
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
        Callable that computes the derivative. If none is provided the average of the
        slope between two points used.
    g: float, defaults = 0.
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase
        difference.
    kwargs: dict
        Extra arguments for custom distance should be put in the kwargs. See the
        documentation for the distance for kwargs.

    Returns
    -------
    float
        Wddtw distance between the two timeseries.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If the metric type cannot be determined
        If the compute derivative callable is not no_python compiled.

    References
    ----------
    .. [1] Young-Seon Jeong, Myong K. Jeong, Olufemi A. Omitaomu, Weighted dynamic time
    warping for time series classification, Pattern Recognition, Volume 44, Issue 9,
    2011, Pages 2231-2240, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2010.09.022.
    """
    format_kwargs = {
        "lower_bounding": lower_bounding,
        "window": window,
        "itakura_max_slope": itakura_max_slope,
        "custom_distance": custom_distance,
        "bounding_matrix": bounding_matrix,
        "compute_derivative": compute_derivative,
        "g": g,
    }
    format_kwargs = {**format_kwargs, **kwargs}

    return distance(x, y, metric="wddtw", **format_kwargs)


def wdtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    custom_distance: DistanceCallable = _SquaredDistance().distance_factory,
    bounding_matrix: np.ndarray = None,
    g: float = 0.0,
    **kwargs: dict,
) -> float:
    """Compute the weighted dynamic time warping (Wdtw) distance between timeseries.

    Wdtw adds a multiplicative weight penalty based on the warping distance between
    points in the warping path. First proposed in [1]_ a weight is applied
    during the distance computation when generating a warping path. This means that
    timeseries with lower phase difference have a smaller weight imposed (i.e less
    penalty imposed) and timeseries with larger phase difference have a larger weight
    imposed (i.e. larger penalty imposed).

    Formally this can be described as:

    .. math::
        d_{w}(x_{i}, y_{j}) = ||w_{|i-j|}(x_{i} - y_{j})||

    Where d_w is the distance with the weight applied to it for points i, j. Where
    w(|i-j|) is a positive weight between the two points x_i and y_j and (x_i - y_j)
    is the distance between x_i and y_j.


    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
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
    g: float, defaults = 0.
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase
        difference.
    kwargs: dict
        Extra arguments for custom distance should be put in the kwargs. See the
        documentation for the distance for kwargs.

    Returns
    -------
    float
        Wdtw distance between the two timeseries.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If the metric type cannot be determined

    References
    ----------
    .. [1] Young-Seon Jeong, Myong K. Jeong, Olufemi A. Omitaomu, Weighted dynamic time
    warping for time series classification, Pattern Recognition, Volume 44, Issue 9,
    2011, Pages 2231-2240, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2010.09.022.
    """
    format_kwargs = {
        "lower_bounding": lower_bounding,
        "window": window,
        "itakura_max_slope": itakura_max_slope,
        "custom_distance": custom_distance,
        "bounding_matrix": bounding_matrix,
        "g": g,
    }
    format_kwargs = {**format_kwargs, **kwargs}

    return distance(x, y, metric="wdtw", **format_kwargs)


def ddtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    custom_distance: DistanceCallable = _SquaredDistance().distance_factory,
    bounding_matrix: np.ndarray = None,
    compute_derivative: DerivativeCallable = _average_of_slope,
    **kwargs: dict,
) -> float:
    r"""Compute the derivative dynamic time warping (Ddtw) distance between timeseries.

    Ddtw distance is supported for 1d, 2d and 3d arrays.

    Ddtw is an adaptation of the original Dtw put forward in forward in [1]_. Ddtw was
    originally put forward in [2]_ and attempts to solve an issue with Dtw in that it
    fails to account for the y axis (or shape) of the timeseries.
    Ddtw attempts to solves this limitation by considering y axis data points as
    higher level features of 'shape'. This is done by taking the first derivative
    of the sequence, and then using this 'derived sequence' to perform a Dtw
    computation. This allows the shape of the timeseries to be considered in
    dtw computation.

    While there are many sophisticated methods for estimating derivatives,
    the average of the slope of the line through the point in question and
    its left neighbour, and the slope of the line through the left neighbour and the
    right neighbour (this can be changed by passing a custom no_python compiled callable
    to compute the derivative via the 'compute_derivative' parameter) is used. See
    [2]_ for explanation.


    Mathematically this derivative is defined as:

    .. math::
        D_{x}[q] = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Where q is the original timeseries and d_q is the derived timeseries.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
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
        Callable that computes the derivative. If none is provided the average of the
        slope between two points used.
    kwargs: dict
        Extra arguments for custom distance should be put in the kwargs. See the
        documentation for the distance for kwargs.

    Returns
    -------
    float
        Ddtw distance between the two timeseries.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric or compute derivative callable is not no_python compiled.
        If the metric type cannot be determined

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
        spoken word recognition," IEEE Transactions on Acoustics, Speech and
        Signal Processing, vol. 26(1), pp. 43--49, 1978.

    .. [2] Keogh, Eamonn & Pazzani, Michael. (2002). Derivative Dynamic Time Warping.
        First SIAM International Conference on Data Mining.
        1. 10.1137/1.9781611972719.1.
    """
    format_kwargs = {
        "lower_bounding": lower_bounding,
        "window": window,
        "itakura_max_slope": itakura_max_slope,
        "custom_distance": custom_distance,
        "bounding_matrix": bounding_matrix,
        "compute_derivative": compute_derivative,
    }
    format_kwargs = {**format_kwargs, **kwargs}

    return distance(x, y, metric="ddtw", **format_kwargs)


def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    custom_distance: DistanceCallable = _SquaredDistance().distance_factory,
    bounding_matrix: np.ndarray = None,
    **kwargs: dict,
) -> float:
    r"""Compute the dynamic time warping (Dtw) distance between two timeseries.

    Dtw distance is supported for 1d, 2d and 3d arrays.

    Originally put forward in [1]_ dtw goal is to compute a more accurate distance
    between two timeseries by considering their alignments during the calculation. This
    is done by measuring the distance (normally using Euclidean) between two timeseries
    and then generate a warping path to 'realign' the two timeseries thereby creating
    a path between the two that accounts for alignment.

    Mathematically dtw can be defined as:

    .. math::
        dtw(x, y) = \sqrt{\sum_{(i, j) \in \pi} \|x_{i} - y_{j}\|^2}


    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
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
        Distance function to used to compute distance between aligned timeseries.
    bounding_matrix: np.ndarray (2d array)
        Custom bounding matrix to use. If defined then other lower_bounding params
        and creation are ignored. The matrix should be structure so that indexes
        considered in bound should be the value 0. and indexes outside the bounding
        matrix should be infinity.
    kwargs: dict
        Extra arguments for custom distance should be put in the kwargs. See the
        documentation for the distance for kwargs.

    Returns
    -------
    float
        Dtw distance between the two timeseries.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """
    format_kwargs = {
        "lower_bounding": lower_bounding,
        "window": window,
        "itakura_max_slope": itakura_max_slope,
        "custom_distance": custom_distance,
        "bounding_matrix": bounding_matrix,
    }
    format_kwargs = {**format_kwargs, **kwargs}

    return distance(x, y, metric="dtw", **format_kwargs)


def squared_distance(x: np.ndarray, y: np.ndarray, **kwargs: dict) -> float:
    r"""Compute the Squared distance between two timeseries.

    Squared distance is supported for 1d, 2d and 3d arrays.

    The squared distance between two timeseries is defined as:

    .. math::
        sd(x, y) = \sum_{i=1}^{n} (x_i - y_i)^2

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d)
        Second timeseries.
    kwargs: dict
        Extra kwargs. For squared there are none however, this is kept for
        consistency.

    Returns
    -------
    float
        Squared distance between the two timeseries.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.
    """
    return distance(x, y, metric="squared", **kwargs)


def euclidean_distance(x: np.ndarray, y: np.ndarray, **kwargs: dict) -> float:
    r"""Compute the Euclidean distance between two timeseries.

    Euclidean distance is supported for 1d, 2d and 3d arrays.

    The euclidean distance between two timeseries is defined as:

    .. math::
        ed(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d)
        Second timeseries.
    kwargs: dict
        Extra kwargs. For euclidean there are none however, this is kept for
        consistency.

    Returns
    -------
    float
        Euclidean distance between the two timeseries.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.
    """
    return distance(x, y, metric="euclidean", **kwargs)


def distance(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[
        str,
        Callable[
            [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
        ],
        Callable[[np.ndarray, np.ndarray], float],
        NumbaDistance,
    ],
    **kwargs: dict,
) -> float:
    """Compute the distance between two timeseries.

    This function works for 1d, 2d and 3d timeseries. No matter how many dimensions
    passed, a single float will always be returned. If you want the distance between
    each timeseries individually look at the pairwise_distance function instead.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
        Second timeseries.
    metric: str or Callable or NumbaDistance
        The distance metric to use.
        If a string is given, the value must be one of the following strings:

        'euclidean', 'squared', 'dtw.

        If callable then it has to be a distance factory or numba distance callable.
        If the distance takes kwargs then a distance factory should be provided. The
        distance factory takes the form:

        Callable[
            [np.ndarray, np.ndarray, bool, dict],
            Callable[[np.ndarray, np.ndarray], float]
        ]

        and should validate the kwargs, and return a no_python callable described
        above as the return.

        If a no_python callable provided it should take the form:

        Callable[
            [np.ndarray, np.ndarray],
            float
        ],
    kwargs: dict, optional
        Extra arguments for metric. Refer to each metric documentation for a list of
        possible arguments.

    Returns
    -------
    float
        Distance between the two timeseries.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.
    """
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)

    _metric_callable = distance_factory(x, y, metric=metric, **kwargs)

    return _compute_distance(_x, _y, _metric_callable)


def distance_factory(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[
        str,
        Callable[
            [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
        ],
        Callable[[np.ndarray, np.ndarray], float],
        NumbaDistance,
    ],
    **kwargs: dict,
) -> DistanceCallable:
    """Create a no_python distance callable.

    This function works for 1d, 2d and 3d timeseries.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
        Second timeseries.
    metric: str or Callable or NumbaDistance
        The distance metric to use.
        If a string is given, the value must be one of the following strings:

        'euclidean', 'squared', 'dtw.

        If callable then it has to be a distance factory or numba distance callable.
        If the distance takes kwargs then a distance factory should be provided. The
        distance factory takes the form:

        Callable[
            [np.ndarray, np.ndarray, bool, dict],
            Callable[[np.ndarray, np.ndarray], float]
        ]

        and should validate the kwargs, and return a no_python callable described
        above as the return.

        If a no_python callable provided it should take the form:

        Callable[
            [np.ndarray, np.ndarray],
            float
        ],
    kwargs: dict, optional
        Extra arguments for metric. Refer to each metric documentation for a list of
        possible arguments.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]]
        No_python compiled distance resolved from the metric input.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.
    """
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)

    return _resolve_metric(metric, _x, _y, _METRIC_INFOS, **kwargs)


def pairwise_distance(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[
        str,
        Callable[
            [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
        ],
        Callable[[np.ndarray, np.ndarray], float],
        NumbaDistance,
    ],
    **kwargs: dict,
) -> np.ndarray:
    """Compute the pairwise distance matrix between two timeseries.

    This function works for 1d, 2d and 3d timeseries. No matter the number of dimensions
    passed a 2d array will always be returned.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
        Second timeseries.
    metric: str or Callable or NumbaDistance
        The distance metric to use.
        If a string is given, the value must be one of the following strings:

        'euclidean', 'squared', 'dtw.

        If callable then it has to be a distance factory or numba distance callable.
        If the distance takes kwargs then a distance factory should be provided. The
        distance factory takes the form:

        Callable[
            [np.ndarray, np.ndarray, bool, dict],
            Callable[[np.ndarray, np.ndarray], float]
        ]

        and should validate the kwargs, and return a no_python callable described
        above as the return.

        If a no_python callable provided it should take the form:

        Callable[
            [np.ndarray, np.ndarray],
            float
        ],
    kwargs: dict, optional
        Extra arguments for metric. Refer to each metric documentation for a list of
        possible arguments.

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y)).
        Pairwise distance matrix between the two timeseries.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.
    """
    _x = to_numba_pairwise_timeseries(x)
    _y = to_numba_pairwise_timeseries(y)
    symmetric = np.array_equal(_x, _y)

    _metric_callable = _resolve_metric(metric, _x, _y, _METRIC_INFOS, **kwargs)
    return _compute_pairwise_distance(_x, _y, symmetric, _metric_callable)


_METRIC_INFOS = [
    MetricInfo(
        canonical_name="erp",
        aka={"erp", "edit distance with real penalty"},
        dist_func=erp_distance,
        dist_instance=_ErpDistance(),
    ),
    MetricInfo(
        canonical_name="edr",
        aka={"edr", "edit distance for real sequences"},
        dist_func=edr_distance,
        dist_instance=_EdrDistance(),
    ),
    MetricInfo(
        canonical_name="lcss",
        aka={"lcss", "longest common subsequence"},
        dist_func=lcss_distance,
        dist_instance=_LcssDistance(),
    ),
    MetricInfo(
        canonical_name="euclidean",
        aka={"euclidean", "ed", "euclid", "pythagorean"},
        dist_func=euclidean_distance,
        dist_instance=_EuclideanDistance(),
    ),
    MetricInfo(
        canonical_name="squared",
        aka={"squared"},
        dist_func=squared_distance,
        dist_instance=_SquaredDistance(),
    ),
    MetricInfo(
        canonical_name="dtw",
        aka={"dtw", "dynamic time warping"},
        dist_func=dtw_distance,
        dist_instance=_DtwDistance(),
    ),
    MetricInfo(
        canonical_name="ddtw",
        aka={"ddtw", "derivative dynamic time warping"},
        dist_func=ddtw_distance,
        dist_instance=_DdtwDistance(),
    ),
    MetricInfo(
        canonical_name="wdtw",
        aka={"wdtw", "weighted dynamic time warping"},
        dist_func=wdtw_distance,
        dist_instance=_WdtwDistance(),
    ),
    MetricInfo(
        canonical_name="wddtw",
        aka={"wddtw", "weighted derivative dynamic time warping"},
        dist_func=wddtw_distance,
        dist_instance=_WddtwDistance(),
    ),
]

_METRICS = {info.canonical_name: info for info in _METRIC_INFOS}
_METRIC_ALIAS = dict((alias, info) for info in _METRIC_INFOS for alias in info.aka)
_METRIC_CALLABLES = dict(
    (info.canonical_name, info.dist_func) for info in _METRIC_INFOS
)
_METRICS_NAMES = list(_METRICS.keys())
