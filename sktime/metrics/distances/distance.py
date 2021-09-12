# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

from typing import Union, Callable, Any, List

import numpy as np

from sktime.metrics.distances.base._types import SktimeSeries, SktimeMatrix
from sktime.metrics.distances.base.base import BaseDistance
from sktime.metrics.distances._scipy_dist import ScipyDistance
from sktime.metrics.distances._squared_dist import SquaredDistance
from sktime.utils.validation.panel import to_numpy_time_series_matrix
from sktime.utils.validation.series import to_numpy_time_series

_distance_registry = {
    "squared": SquaredDistance,
    "braycurtis": ScipyDistance,
    "canberra": ScipyDistance,
    "chebyshev": ScipyDistance,
    "cityblock": ScipyDistance,
    "correlation": ScipyDistance,
    "cosine": ScipyDistance,
    "dice": ScipyDistance,
    "euclidean": ScipyDistance,
    "hamming": ScipyDistance,
    "jaccard": ScipyDistance,
    "jensenshannon": ScipyDistance,
    "kulsinski": ScipyDistance,
    # "mahalanobis": ScipyDistance, # This isn't working currently not sure why
    "matching": ScipyDistance,
    "minkowski": ScipyDistance,
    "rogerstanimoto": ScipyDistance,
    "russellrao": ScipyDistance,
    "seuclidean": ScipyDistance,
    "sokalmichener": ScipyDistance,
    "sokalsneath": ScipyDistance,
    "sqeuclidean": ScipyDistance,
    "yule": ScipyDistance,
}


def _resolve_metric(metric: str, **kwargs: Any) -> BaseDistance:
    """
    Method used to resolve the passed distance metric

    Parameters
    ----------
    metric: str
        Metric to use to perform distance computation
    kwargs: Any
        Kwargs to be passed to the distance metric

    Returns
    -------
    BaseDistance
        Created distance object to perform the distance computation
    """
    if metric in _distance_registry:
        if metric in ScipyDistance.supported_scipy_distances():
            return _distance_registry[metric](metric, **kwargs)
        return _distance_registry[metric](**kwargs)
    else:
        raise ValueError(
            "The str provided is not a registered distance. Please check"
            "the distance specified is valid. To see a list of available"
            "distances, call 'get_available_distances()'"
        )


def distance(
    x: SktimeSeries,
    y: SktimeSeries,
    metric: Union[str, BaseDistance, Callable] = 'euclidean',
    **kwargs: Any
) -> float:
    """
    Method used to get the distance between two time series

    Parameters
    ----------
    x: np.ndarray or pd.DataFrame or pd.Series or List
        First time series
    y: np.ndarray or pd.DataFrame or pd.Series or List
        Second time series
    metric: str or BaseDistance or Callable, defaults = 'euclidean'
        Metric to use to perform distance computation
    kwargs: Any
        Kwargs to be passed to the distance metric

    Returns
    -------
    float
        Distance between the two time series
    """
    if isinstance(metric, str):
        return _resolve_metric(metric, **kwargs).distance(x, y)
    elif isinstance(metric, BaseDistance):
        return metric.distance(x, y)
    else:
        x = to_numpy_time_series(x)
        y = to_numpy_time_series(y)
        if x.ndim != y.ndim:
            raise ValueError(
                "The number of dims of x must match the number of" "dims of y"
            )
        return metric(x, y, **kwargs)


def pairwise(
    x: SktimeMatrix,
    y: SktimeMatrix = None,
    metric: Union[str, BaseDistance, Callable] = 'euclidean',
    **kwargs: Any
) -> np.ndarray:
    """
    Method to compute a pairwise distance on a matrix (i.e. distance between each
    ts in the matrix)

    Parameters
    ----------
    x: np.ndarray or pd.Dataframe or List
        First matrix of multiple time series
    y: np.ndarray or pd.Dataframe or List, defaults = x
        Second matrix of multiple time series.
    metric: str or BaseDistance or Callable, defaults = 'euclidean'
        Metric to use to perform distance computation
    kwargs: Any
        Kwargs to be passed to the distance metric

    Returns
    -------
    np.ndarray
        Matrix containing the pairwise distance between each point
    """
    if y is None:
        y = x

    if isinstance(metric, str):
        return _resolve_metric(metric, **kwargs).pairwise(x, y)
    elif isinstance(metric, BaseDistance):
        return metric.pairwise(x, y)
    else:
        x, y, symmetric = BaseDistance.format_pairwise_matrix(x, y)
        return metric(x, y, **kwargs)


def get_available_distances() -> List[str]:
    """
    Method used to get all the available distances in string format

    Returns
    -------
    List[str]
        List of string value metrics
    """
    return list(_distance_registry.keys())
