# -*- coding: utf-8 -*-
"""Compute the distance between two timeseries."""

import inspect
from typing import Callable, NamedTuple, Set, Union

import numpy as np
from numba import njit

from sktime.dists_kernels._utils import to_numba_timeseries
from sktime.dists_kernels.numba.distances._euclidean_distance import _EuclideanDistance
from sktime.dists_kernels.numba.distances._squared_distance import _SquaredDistance
from sktime.dists_kernels.numba.distances.base import DistanceCallable, NumbaDistance


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
    distance: float
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
    distance: float
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

    _metric_callable = _resolve_metric(metric, _x, _y, **kwargs)

    return _compute_distance(_x, _y, _metric_callable)


@njit(cache=True)
def _compute_distance(
    x: np.ndarray, y: np.ndarray, distance_callable: DistanceCallable
) -> float:
    """Compute distance between two 3d numpy array.

    Parameters
    ----------
    x: np.ndarray (3d array)
        First timeseries.
    y: np.ndarray (3d array)
        Second timeseries.
    distance_callable: Callable[[np.ndarray, np.ndarray], float]
        No_python distance callable to measure the distance between two 2d numpy
        arrays.

    Returns
    -------
    float
        Distance between two timeseries.
    """
    loop_to: int = min(x.shape[0], y.shape[0])

    total_distance = 0.0

    for i in range(loop_to):
        total_distance += distance_callable(x[i], y[i])

    return total_distance


def _resolve_metric(
    metric: Union[str, Callable, NumbaDistance],
    x: np.ndarray,
    y: np.ndarray,
    **kwargs: dict,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Resolve a metric from a string or callable or NumbaDistance instance.

    Parameters
    ----------
    metric: str or Callable or NumbaDistance
        The distance metric to use.
    x: np.ndarray (3d array)
        First timeseries.
    y: np.ndarray (3d array)
        Second timeseries.
    kwargs: dict, optional
        Extra arguments for metric. Refer to each metric documentation for a list of
        possible arguments.


    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]]
        Distance metric resolved from the metric input.

    Raises
    ------
    ValueError
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.
    """
    numba_dist_instance: Union[NumbaDistance, None] = None
    if isinstance(metric, NumbaDistance):
        numba_dist_instance = metric
    elif isinstance(metric, str):
        numba_dist_instance = _resolve_str_metric(metric)
    elif callable(metric):
        if _is_distance_factory_callable(metric):
            metric = metric(x[0], y[0], **kwargs)
        elif _is_no_python_distance_callable(metric):
            metric = metric
        else:
            raise ValueError(
                "The callable provided must be no_python (using njit()) for"
                "this operation. Please compile the function and try again."
            )
    else:
        raise ValueError(
            "Unable to resolve the metric with the parameters provided."
            "The metric must be a valid string, NumbaDistance or a"
            "distance factory callable or no_python distance."
        )

    if numba_dist_instance is not None:
        metric = numba_dist_instance.distance_factory(x[0], y[0], **kwargs)

    return metric


def _resolve_str_metric(metric: str) -> NumbaDistance:
    """Resolve a string to a NumbaDistance.

    Parameters
    ----------
    metric: str
        String to resolve to NumbaDistance.

    Returns
    -------
    NumbaDistance
        Instance of distance resolved from string

    Raises
    ------
    ValueError
        If the metric string provided is not a known distance.
    """
    metric_info: MetricInfo = _METRIC_ALIAS.get(metric, None)
    if metric_info is None:
        raise ValueError(
            f"The metric provided: {metric}, is invalid. The current list"
            f"of supported distances is {_METRICS_NAMES}"
        )
    return metric_info.dist_instance


def _is_distance_factory_callable(metric: Callable) -> bool:
    """Validate if a callable is a distance factory.

    Parameters
    ----------
    metric: Callable
        Callable to validate if is a valid distance factory.

    Returns
    -------
    bool
        Boolean that is true if callable is a valid distance factory and false
        if the callable is an invalid distance factory.
    """
    is_no_python_compiled = hasattr(metric, "signatures")
    if is_no_python_compiled:
        return False
    correct_num_params = len(inspect.signature(metric).parameters) >= 2
    return_num_params = len(inspect.signature(metric).return_annotation) == 1
    return correct_num_params and return_num_params


def _is_no_python_distance_callable(metric: Callable) -> bool:
    """Validate if a callable is a no_python compiled distance metric.

    Parameters
    ----------
    metric: Callable
        Callable to validate if is a valid no_python distance callable.

    Returns
    -------
    bool
        Boolean that is true if callable is a valid no_python compiled distance and
        false if the callable is an invalid no_python callable.

    """
    is_no_python_compiled = hasattr(metric, "signatures")
    if not is_no_python_compiled:
        return False
    correct_num_params = len(inspect.signature(metric).parameters) == 2
    return_num_params = inspect.signature(metric).return_annotation is float
    return correct_num_params and return_num_params


class MetricInfo(NamedTuple):
    """Define a registry entry for a metric."""

    # Name of the distance
    canonical_name: str
    # All aliases, including canonical_name
    aka: Set[str]
    # Python distance function (can use numba inside but callable must be in python)
    dist_func: Callable
    # NumbaDistance class
    dist_instance: NumbaDistance


# Registry of implemented metrics:
_METRIC_INFOS = [
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
]

_METRICS = {info.canonical_name: info for info in _METRIC_INFOS}
_METRIC_ALIAS = dict((alias, info) for info in _METRIC_INFOS for alias in info.aka)

_METRICS_NAMES = list(_METRICS.keys())
