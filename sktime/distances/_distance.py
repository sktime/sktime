# -*- coding: utf-8 -*-
"""Compute the distance between two timeseries."""

from typing import Any, Callable, Union

import numpy as np

from sktime.distances._ddtw import DerivativeCallable, _average_of_slope, _DdtwDistance
from sktime.distances._dtw import _DtwDistance
from sktime.distances._edr import _EdrDistance
from sktime.distances._erp import _ErpDistance
from sktime.distances._euclidean import _EuclideanDistance
from sktime.distances._lcss import _LcssDistance
from sktime.distances._numba_utils import (
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
    **kwargs: Any,
) -> float:
    """Compute the Edit distance for real penalty (erp) distance between two series.

    Erp first proposed in [1]_ attempts to improve accuracy of distance computation
    by better considering how indexes are carried forward through the cost matrix.
    Usually in the dtw cost matrix, if an alignment can't be found the previous value
    is carried forward. Erp instead proposes the idea of gaps or sequences of points
    that have no matches. These gaps are then punished based on their distance from 'g'.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First timeseries.
    y: np.ndarray (1d or 2d array)
        Second timeseries.
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        Lower bounding technique to use.
        If LowerBounding enum provided, the following are valid:
            LowerBounding.NO_BOUNDING - No bounding
            LowerBounding.SAKOE_CHIBA - Sakoe chiba
            LowerBounding.ITAKURA_PARALLELOGRAM - Itakura parallelogram
        If int value provided, the following are valid:
            1 - No bounding
            2 - Sakoe chiba
            3 - Itakura parallelogram
    window: int, defaults = 2
        Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding).
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura parallelogram (if using Itakura
        Parallelogram lower bounding).
    custom_distance: str or Callable, defaults = euclidean
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y)),
                                    defaults = None)
        Custom bounding matrix to use. If defined then other lower_bounding params
        are ignored. The matrix should be structure so that indexes considered in
        bound should be the value 0. and indexes outside the bounding matrix should be
        infinity.
    g: float, defaults = 0.
        The reference value to penalise gaps.
    kwargs: Any
        Extra arguments for custom distances. See the documentation for the
        distance itself for valid kwargs.

    Returns
    -------
    float
        Erp distance between x and y.

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
        If g is not a float.

    Examples
    --------
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> erp_distance(x_1d, y_1d)
    16.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> erp_distance(x_2d, y_2d)
    32.0

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
    custom_distance: DistanceCallable = _EuclideanDistance().distance_factory,
    bounding_matrix: np.ndarray = None,
    epsilon: float = None,
    **kwargs: Any,
) -> float:
    """Compute the Edit distance for real sequences (edr) between two series.

    Edr computes the minimum number of elements (as a percentage) that must be removed
    from x and y so that the sum of the distance between the remaining signal elements
    lies within the tolerance (epsilon). Edr was originally put forward in [1]_.

    The value returned will be between 0 and 1 per time series. The value will
    represent as a percentage of elements that must be removed for the timeseries to
    be an exact match.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First timeseries.
    y: np.ndarray (1d or 2d array)
        Second timeseries.
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        Lower bounding technique to use.
        If LowerBounding enum provided, the following are valid:
            LowerBounding.NO_BOUNDING - No bounding
            LowerBounding.SAKOE_CHIBA - Sakoe chiba
            LowerBounding.ITAKURA_PARALLELOGRAM - Itakura parallelogram
        If int value provided, the following are valid:
            1 - No bounding
            2 - Sakoe chiba
            3 - Itakura parallelogram
    window: int, defaults = 2
        Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding).
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura parallelogram (if using Itakura
        Parallelogram lower bounding).
    custom_distance: str or Callable, defaults = euclidean
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict],
            Callable[[np.ndarray, np.ndarray], float]]
    bounding_matrix: np.ndarray (2d array)
        Custom bounding matrix to use. If defined then other lower_bounding params
        are ignored. The matrix should be structure so that indexes considered in
        bound should be the value 0. and indexes outside the bounding matrix should be
        infinity.
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.
    kwargs: Any
        Extra arguments for custom distance should be put in the kwargs. See the
        documentation for the distance for kwargs.

    Returns
    -------
    float
        Edr distance between the x and y. The value will be between 0.0 and 1.0
        where 0.0 is an exact match between timeseries (i.e. they are the same) and
        1.0 where there are no matching subsequences.

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

    Examples
    --------
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> edr_distance(x_1d, y_1d)
    1.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> edr_distance(x_2d, y_2d)
    1.0

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
    custom_distance: DistanceCallable = _EuclideanDistance().distance_factory,
    bounding_matrix: np.ndarray = None,
    epsilon: float = 1.0,
    **kwargs: Any,
) -> float:
    """Compute the longest common subsequence (lcss) score between two timeseries.

    Lcss attempts to find the longest common sequence between two timeseries and returns
    a value that is the percentage that longest common sequence assumes. Originally
    present in [1]_, lcss is computed by matching indexes that are similar up until a
    defined threshold (epsilon).

    The value returned will be between 0.0 and 1.0, where 0.0 means the two timeseries
    are exactly the same and 1.0 means they are complete opposites.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First timeseries.
    y: np.ndarray (1d or 2d array)
        Second timeseries.
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        Lower bounding technique to use.
        If LowerBounding enum provided, the following are valid:
            LowerBounding.NO_BOUNDING - No bounding
            LowerBounding.SAKOE_CHIBA - Sakoe chiba
            LowerBounding.ITAKURA_PARALLELOGRAM - Itakura parallelogram
        If int value provided, the following are valid:
            1 - No bounding
            2 - Sakoe chiba
            3 - Itakura parallelogram
    window: int, defaults = 2
        Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding).
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura parallelogram (if using Itakura
        Parallelogram lower bounding).
    custom_distance: str or Callable, defaults = euclidean
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y)),
                                    defaults = None)
        Custom bounding matrix to use. If defined then other lower_bounding params
        are ignored. The matrix should be structure so that indexes considered in
        bound should be the value 0. and indexes outside the bounding matrix should be
        infinity.
    epsilon : float, defaults = 1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'.
    kwargs: Any
        Extra arguments for custom distance should be put in the kwargs. See the
        documentation for the distance for kwargs.

    Returns
    -------
    float
        Lcss distance between x and y. The value returned will be between 0.0 and 1.0,
        where 0.0 means the two timeseries are exactly the same and 1.0 means they
        are complete opposites.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If the metric type cannot be determined

    Examples
    --------
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> lcss_distance(x_1d, y_1d)
    1.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> lcss_distance(x_2d, y_2d)
    1.0

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
    **kwargs: Any,
) -> float:
    r"""Compute the weighted derivative dynamic time warping (wddtw) distance.

    Wddtw was first proposed in [1]_ as an extension of ddtw. By adding a weight
    to the derivative it means the alignment isn't only considering the shape of the
    timeseries, but also the phase.

    Formally the derivative is calculated as:

    .. math::
        D_{x}[q] = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Therefore a weighted derivative can be calculated using D (the derivative) as:

    .. math::
        d_{w}(x_{i}, y_{j}) = ||w_{|i-j|}(D_{x_{i}} - D_{y_{j}})||

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First timeseries.
    y: np.ndarray (1d or 2d array)
        Second timeseries.
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        Lower bounding technique to use.
        If LowerBounding enum provided, the following are valid:
            LowerBounding.NO_BOUNDING - No bounding
            LowerBounding.SAKOE_CHIBA - Sakoe chiba
            LowerBounding.ITAKURA_PARALLELOGRAM - Itakura parallelogram
        If int value provided, the following are valid:
            1 - No bounding
            2 - Sakoe chiba
            3 - Itakura parallelogram
    window: int, defaults = 2
        Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding).
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura parallelogram (if using Itakura
        Parallelogram lower bounding).
    custom_distance: str or Callable, defaults = squared euclidean
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y)),
                                    defaults = None)
        Custom bounding matrix to use. If defined then other lower_bounding params
        are ignored. The matrix should be structure so that indexes considered in
        bound should be the value 0. and indexes outside the bounding matrix should be
        infinity.
    compute_derivative: Callable[[np.ndarray], np.ndarray],
                            defaults = average slope difference
        Callable that computes the derivative. If none is provided the average of the
        slope between two points used.
    g: float, defaults = 0.
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase
        difference.
    kwargs: Any
        Extra arguments for custom distances. See the documentation for the
        distance itself for valid kwargs.

    Returns
    -------
    float
        Wddtw distance between x and y.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If the metric type cannot be determined
        If the compute derivative callable is not no_python compiled.
        If the value of g is not a float

    Examples
    --------
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> wddtw_distance(x_1d, y_1d)
    0.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> wddtw_distance(x_2d, y_2d)
    0.0

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
    **kwargs: Any,
) -> float:
    """Compute the weighted dynamic time warping (wdtw) distance between timeseries.

    First proposed in [1]_, wdtw adds a  adds a multiplicative weight penalty based on
    the warping distance. This means that timeseries with lower phase difference have
    a smaller weight imposed (i.e less penalty imposed) and timeseries with larger phase
    difference have a larger weight imposed (i.e. larger penalty imposed).

    Formally this can be described as:

    .. math::
        d_{w}(x_{i}, y_{j}) = ||w_{|i-j|}(x_{i} - y_{j})||

    Where d_w is the distance with a the weight applied to it for points i, j, where
    w(|i-j|) is a positive weight between the two points x_i and y_j.


    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First timeseries.
    y: np.ndarray (1d or 2d array)
        Second timeseries.
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        Lower bounding technique to use.
        If LowerBounding enum provided, the following are valid:
            LowerBounding.NO_BOUNDING - No bounding
            LowerBounding.SAKOE_CHIBA - Sakoe chiba
            LowerBounding.ITAKURA_PARALLELOGRAM - Itakura parallelogram
        If int value provided, the following are valid:
            1 - No bounding
            2 - Sakoe chiba
            3 - Itakura parallelogram
    window: int, defaults = 2
        Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding).
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura parallelogram (if using Itakura
        Parallelogram lower bounding).
    custom_distance: str or Callable, defaults = squared euclidean
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y)),
                                    defaults = None)
        Custom bounding matrix to use. If defined then other lower_bounding params
        are ignored. The matrix should be structure so that indexes considered in
        bound should be the value 0. and indexes outside the bounding matrix should be
        infinity.
    g: float, defaults = 0.
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase
        difference.
    kwargs: Any
        Extra arguments for custom distances. See the documentation for the
        distance itself for valid kwargs.

    Returns
    -------
    float
        Wdtw distance between the x and y.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If the metric type cannot be determined

    Examples
    --------
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> wdtw_distance(x_1d, y_1d)
    5.385164807134504

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> wdtw_distance(x_2d, y_2d)
    16.0

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
    **kwargs: Any,
) -> float:
    r"""Compute the derivative dynamic time warping (ddtw) distance between timeseries.

    Ddtw is an adaptation of Dtw originally proposed in [1]_. Ddtw attempts to
    improve on dtw by better account for the 'shape' of the timeseries.
    This is done by considering y axis data points as higher level features of 'shape'.
    To do this the first derivative of the sequence is taken, and then using this
    derived sequence a dtw computation is done.

    The default derivative used is:

    .. math::
        D_{x}[q] = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Where q is the original timeseries and d_q is the derived timeseries.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First timeseries.
    y: np.ndarray (1d or 2d array)
        Second timeseries.
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        Lower bounding technique to use.
        If LowerBounding enum provided, the following are valid:
            LowerBounding.NO_BOUNDING - No bounding
            LowerBounding.SAKOE_CHIBA - Sakoe chiba
            LowerBounding.ITAKURA_PARALLELOGRAM - Itakura parallelogram
        If int value provided, the following are valid:
            1 - No bounding
            2 - Sakoe chiba
            3 - Itakura parallelogram
    window: int, defaults = 2
        Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding).
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura parallelogram (if using Itakura
        Parallelogram lower bounding).
    custom_distance: str or Callable, defaults = squared euclidean
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y)),
                                    defaults = None)
        Custom bounding matrix to use. If defined then other lower_bounding params
        are ignored. The matrix should be structure so that indexes considered in
        bound should be the value 0. and indexes outside the bounding matrix should be
        infinity.
    compute_derivative: Callable[[np.ndarray], np.ndarray],
                            defaults = average slope difference
        Callable that computes the derivative. If none is provided the average of the
        slope between two points used.
    kwargs: Any
        Extra arguments for custom distances. See the documentation for the
        distance itself for valid kwargs.

    Returns
    -------
    float
        Ddtw distance between the x and y.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric or compute derivative callable is not no_python compiled.
        If the metric type cannot be determined
        If the compute derivative callable is not no_python compiled.

    Examples
    --------
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> ddtw_distance(x_1d, y_1d)
    0.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> ddtw_distance(x_2d, y_2d)
    0.0

    References
    ----------
    .. [1] Keogh, Eamonn & Pazzani, Michael. (2002). Derivative Dynamic Time Warping.
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
    **kwargs: Any,
) -> float:
    r"""Compute the dynamic time warping (dtw) distance between two timeseries.

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
    x: np.ndarray (1d or 2d array)
        First timeseries.
    y: np.ndarray (1d or 2d array)
        Second timeseries.
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        Lower bounding technique to use.
        If LowerBounding enum provided, the following are valid:
            LowerBounding.NO_BOUNDING - No bounding
            LowerBounding.SAKOE_CHIBA - Sakoe chiba
            LowerBounding.ITAKURA_PARALLELOGRAM - Itakura parallelogram
        If int value provided, the following are valid:
            1 - No bounding
            2 - Sakoe chiba
            3 - Itakura parallelogram
    window: int, defaults = 2
        Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding).
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura parallelogram (if using Itakura
        Parallelogram lower bounding).
    custom_distance: str or Callable, defaults = squared euclidean
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y)),
                                    defaults = None)
        Custom bounding matrix to use. If defined then other lower_bounding params
        are ignored. The matrix should be structure so that indexes considered in
        bound should be the value 0. and indexes outside the bounding matrix should be
        infinity.
    kwargs: Any
        Extra arguments for custom distance should be put in the kwargs. See the
        documentation for the distance for kwargs.

    Returns
    -------
    float
        Dtw distance between x and y.

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
        If the itakura_max_slope is not a float or int.
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined

    Examples
    --------
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> dtw_distance(x_1d, y_1d)
    58.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> dtw_distance(x_2d, y_2d)
    512.0

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


def squared_distance(x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
    r"""Compute the Squared distance between two timeseries.

    The squared distance between two timeseries is defined as:

    .. math::
        sd(x, y) = \sum_{i=1}^{n} (x_i - y_i)^2

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First timeseries.
    y: np.ndarray (1d or 2d array)
        Second timeseries.
    kwargs: Any
        Extra kwargs. For squared there are none however, this is kept for
        consistency.

    Returns
    -------
    float
        Squared distance between x and y.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.

    Examples
    --------
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> squared_distance(x_1d, y_1d)
    64.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> squared_distance(x_2d, y_2d)
    512.0
    """
    return distance(x, y, metric="squared", **kwargs)


def euclidean_distance(x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
    r"""Compute the Euclidean distance between two timeseries.

    Euclidean distance is supported for 1d, 2d and 3d arrays.

    The euclidean distance between two timeseries is defined as:

    .. math::
        ed(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First timeseries.
    y: np.ndarray (1d or 2d array)
        Second timeseries.
    kwargs: Any
        Extra kwargs. For euclidean there are none however, this is kept for
        consistency.

    Returns
    -------
    float
        Euclidean distance between x and y.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.

    Examples
    --------
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> euclidean_distance(x_1d, y_1d)
    8.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> euclidean_distance(x_2d, y_2d)
    22.627416997969522
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
    **kwargs: Any,
) -> float:
    """Compute the distance between two timeseries.

    First the distance metric is 'resolved'. This means the metric that is passed
    is resolved to a callable. The callable is then called with x and y and the
    value is then returned.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First timeseries.
    y: np.ndarray (1d or 2d array)
        Second timeseries.
    metric: str or Callable
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    kwargs: Any
        Arguments for metric. Refer to each metrics documentation for a list of
        possible arguments.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.

    Examples
    --------
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> distance(x_1d, y_1d, metric='dtw')
    58.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> distance(x_2d, y_2d, metric='dtw')
    512.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> distance(x_2d, y_2d, metric='dtw', lower_bounding=2, window=3)
    512.0

    Returns
    -------
    float
        Distance between the x and y.
    """
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)

    _metric_callable = _resolve_metric(metric, _x, _y, _METRIC_INFOS, **kwargs)

    return _metric_callable(_x, _y)


def distance_factory(
    x: np.ndarray = None,
    y: np.ndarray = None,
    metric: Union[
        str,
        Callable[
            [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
        ],
        Callable[[np.ndarray, np.ndarray], float],
        NumbaDistance,
    ] = "euclidean",
    **kwargs: Any,
) -> DistanceCallable:
    """Create a no_python distance callable.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array), defaults = None
        First timeseries.
    y: np.ndarray (1d or 2d array), defaults = None
        Second timeseries.
    metric: str or Callable, defaults  = 'euclidean'
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    kwargs: Any
        Arguments for metric. Refer to each metrics documentation for a list of
        possible arguments.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]]
        No_python compiled distance callable.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.
    """
    if x is None:
        x = np.zeros((10, 10))
    if y is None:
        y = np.zeros((10, 10))
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
    **kwargs: Any,
) -> np.ndarray:
    """Compute the pairwise distance matrix between two timeseries.

    First the distance metric is 'resolved'. This means the metric that is passed
    is resolved to a callable. The callable is then called with x and y and the
    value is then returned. Then for each combination of x and y, the distance between
    the values are computed resulting in a 2d pairwise matrix.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
        Second timeseries.
    metric: str or Callable
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    kwargs: Any
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

    Examples
    --------
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> pairwise_distance(x_1d, y_1d, metric='dtw')
    array([[64.]])

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> pairwise_distance(x_2d, y_2d, metric='dtw')
    array([[256., 576.],
           [ 58., 256.]])

    >>> x_3d = np.array([[[1], [2], [3], [4]], [[5], [6], [7], [8]]])  # 3d array
    >>> y_3d = np.array([[[9], [10], [11], [12]], [[13], [14], [15], [16]]])  # 3d array
    >>> pairwise_distance(x_3d, y_3d, metric='dtw')
    array([[256., 576.],
           [ 58., 256.]])

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> pairwise_distance(x_2d, y_2d, metric='dtw', lower_bounding=2, window=3)
    array([[256., 576.],
           [ 58., 256.]])
    """
    _x = to_numba_pairwise_timeseries(x)
    _y = to_numba_pairwise_timeseries(y)
    symmetric = np.array_equal(_x, _y)

    _metric_callable = _resolve_metric(metric, _x[0], _y[0], _METRIC_INFOS, **kwargs)
    return _compute_pairwise_distance(_x, _y, symmetric, _metric_callable)


_METRIC_INFOS = [
    MetricInfo(
        canonical_name="euclidean",
        aka={"euclidean", "ed", "euclid", "pythagorean"},
        dist_func=euclidean_distance,
        dist_instance=_EuclideanDistance(),
    ),
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
