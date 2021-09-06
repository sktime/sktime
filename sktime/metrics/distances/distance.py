# -*- coding: utf-8 -*-
from typing import Union, Callable, Any, List

import numpy as np

from sktime.metrics.distances.base._types import SktimeSeries, SktimeMatrix
from sktime.metrics.distances.base.base import BaseDistance
from sktime.metrics.distances._scipy_dist import ScipyDistance
from sktime.metrics.distances._squared_dist import SquaredDistance
from sktime.metrics.distances.dtw._dtw import Dtw
from sktime.metrics.distances.dtw._fast_dtw import FastDtw
from sktime.metrics.distances.dtw._dtw_path import DtwPath
from sktime.metrics.distances.dtw._dtw_cost_matrix import DtwCostMatrix

_distance_registry = {
    "dtw": Dtw,
    "dtw_path": DtwPath,
    "dtw_cost_metric": DtwCostMatrix,
    "fast_dtw": FastDtw,
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
    # "mahalanobis": ScipyDistance,
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
    metric: Union[str, BaseDistance, Callable],
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
    metric: str or BaseDistance or Callable
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
        return metric(x, y, **kwargs)


def pairwise(
    x: SktimeMatrix,
    y: SktimeMatrix,
    metric: Union[str, BaseDistance, Callable],
    **kwargs: Any
) -> np.ndarray:
    """
    Method to compute a pairwise distance on a matrix (i.e. distance between each
    ts in the matrix)

    Parameters
    ----------
    x: np.ndarray or pd.Dataframe or List
        First matrix of multiple time series
    y: np.ndarray or pd.Dataframe or List
        Second matrix of multiple time series.
    metric: str or BaseDistance or Callable
        Metric to use to perform distance computation
    kwargs: Any
        Kwargs to be passed to the distance metric

    Returns
    -------
    np.ndarray
        Matrix containing the pairwise distance between each point
    """

    if isinstance(metric, str):
        return _resolve_metric(metric, **kwargs).pairwise(x, y)
    elif isinstance(metric, BaseDistance):
        return metric.pairwise(x, y)
    else:
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
