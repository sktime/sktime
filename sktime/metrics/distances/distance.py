# -*- coding: utf-8 -*-
from typing import Union, Callable, Any, Optional, List, Set
import dataclasses
import numpy as np
from scipy.spatial.distance import cdist, pdist
from numba import njit, prange

from sktime.metrics.distances import dtw, dtw_and_cost_matrix

_SCIPY_DISTS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulsinski",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]


@dataclasses.dataclass(frozen=True)
class MetricInfo:
    # Name of python distance function
    canonical_name: str
    # All aliases, including canonical_name
    aka: Set[str]
    # unvectorized distance function
    dist_func: Callable
    # dist func without checks of kwargs and series checks. Assumes all arguments are
    # valid.
    optimised_dist_func: Callable
    # Validator for arguments for the dist function
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
        optimised_dist_func=cdist,
    ),
]

_METRICS = {info.canonical_name: info for info in _METRIC_INFOS}
_METRIC_ALIAS = dict((alias, info) for info in _METRIC_INFOS for alias in info.aka)
_METRICS_NAMES = list(_METRICS.keys())
