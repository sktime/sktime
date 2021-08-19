# -*- coding: utf-8 -*-
from typing import Union, Callable, Any, Optional, List, Set
import dataclasses
import numpy as np
from scipy.spatial.distance import cdist, pdist
from numba import njit, prange

from sktime.metrics.distances import dtw, dtw_and_cost_matrix
from sktime.metrics.distances._dtw_based import (
    _dtw,
    _dtw_and_cost_matrix,
    _dtw_check_params,
)
from sktime.metrics.distances._distance_utils import (
    format_distance_series,
    format_pairwise_matrix,
    SktimeMatrix,
    SktimeSeries,
)


@dataclasses.dataclass(frozen=True)
class MetricInfo:
    # Name of python distance function
    canonical_name: str
    # All aliases, including canonical_name
    aka: Set[str]
    # unvectorized distance function
    dist_func: Callable
    # Optimized cdist function
    cdist_func: Callable
    # Optimized pdist function
    pdist_func: Callable
    # function that checks kwargs and computes default values:
    # f(X, m, n, **kwargs)
    validator: Optional[Callable] = None
    # list of supported types:
    # X (pdist) and XA (cdist) are used to choose the type. if there is no
    # match the first type is used. Default double
    types: List[str] = dataclasses.field(default_factory=lambda: ["double"])


# Registry of implemented metrics:
_METRIC_INFOS = [
    MetricInfo(
        canonical_name="scipy",
        aka={"scipy cdist", "scipy pdist"},
        dist_func=cdist,
        cdist_func=cdist,
        pdist_func=pdist,
    ),
    MetricInfo(
        canonical_name="dtw",
        aka={"dtw", "dynamic time warping"},
        dist_func=dtw,
        cdist_func=_dtw,
        pdist_func=_dtw,
        validator=_dtw_check_params,
    ),
]

_METRICS = {info.canonical_name: info for info in _METRIC_INFOS}
_METRIC_ALIAS = dict((alias, info) for info in _METRIC_INFOS for alias in info.aka)

_METRICS_NAMES = list(_METRICS.keys())


def _create_metric_wrapper(metric_info: MetricInfo, is_cdist: bool) -> Callable:
    """
    Method used to produce a callable distance function from a str

    Parameters
    ----------
    metric_info: MetricInfo
        Info containing methods for the distance metric
    is_cdist: bool
        When True cdist function retrieved, when False pdist function retrieved

    Returns
    -------
    Callable
        Callable for the distance method containing validator and call to distance

    """
    func_key = "cdist_func"
    if not is_cdist:
        func_key = "pdist_func"

    validator = metric_info.validator

    callable_dist = metric_info[func_key]

    def distance_func(x, y, **kwargs):
        if validator is not None:
            kwargs = validator(kwargs)
        return callable_dist(x, y, kwargs)

    return distance_func


def _resolve_metric(metric: Union[str, Callable], is_cdist: bool = True) -> Callable:
    """
    Method that is used to resolve a metric callable or str

    Parameters
    ----------
    metric: str or Callable
        Metric to resolve
    is_cdist: bool, defaults = True
        When True if metric is a str the cdist distance function will be used
        and when False if is a str the pdist distance function will be used

    Returns
    -------
    Callable
        Function that has been created containing a call to the validator (where exists)
        and the distance function. This is then ready to be called by pdist or cdist
    """
    if isinstance(metric, Callable):
        return metric
    elif isinstance(metric, str):
        if metric in _METRICS:
            # One of the sktime distances
            return _create_metric_wrapper(_METRICS[metric], is_cdist)
        else:
            # Assume it is a scipy distance
            return _create_metric_wrapper(_METRICS["scipy"], is_cdist)
    else:
        raise ValueError(
            "The metric parameter passed is invalid. It must be either a"
            "Callable or str"
        )


def ts_cdist(
    x: SktimeSeries,
    y: SktimeSeries,
    metric: Union[str, Callable] = "dtw",
    **kwargs: Any
) -> np.ndarray:
    x, y = format_distance_series(x, y)

    callable_dist = _resolve_metric(metric, True)

    return callable_dist(x, y, **kwargs)


@njit(parallel=True)
def _njit_pdist(callable_dist: Callable, x: np.ndarray, **kwargs) -> np.ndarray:
    x_size = x.shape[0]
    dist_matrix = np.zeros((x_size, x_size))
    for i in prange(x_size):
        for j in range(x_size):
            dist_matrix[i, j] = callable_dist(x, **kwargs)

    return dist_matrix


def ts_pdist(
    x: SktimeMatrix, metric: Union[str, Callable] = "dtw", **kwargs: Any
) -> np.ndarray:
    x = format_pairwise_matrix(x)

    callable_dist = _resolve_metric(metric, False)

    return callable_dist(x, **kwargs)
