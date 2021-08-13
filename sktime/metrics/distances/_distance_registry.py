# -*- coding: utf-8 -*-
__all__ = ["DISTANCE_INFO"]

from typing import Callable, Set, Type, Optional
from dataclasses import dataclass

from sktime.metrics.distances.base._base import BaseDistance, BasePairwise
from sktime.metrics.distances._dtw import _DtwDistance, _DtwDistanceCostMatrix


@dataclass(frozen=True)
class DistanceInfo:
    # Name of python distance function
    canonical_name: str
    # All aliases, including canonical_name
    aka: Set[str]
    # Base distance class
    base_distance_class: Optional[Type[BaseDistance]]
    # Base pairwise class
    base_pairwise_class: Optional[Type[BasePairwise]]
    # Distance function to call
    dist_func: Callable
    # Pairwise function to call
    pairwise_dist_func: Optional[Callable]


# Registry of implemented metrics:
DISTANCE_INFO = [
    DistanceInfo(
        canonical_name="dtw",
        aka={"dtw", "dynamic time warping"},
        base_distance_class=_DtwDistance,
        dist_func=_DtwDistance().distance,
        pairwise_dist_func=_DtwDistance().distance,
    ),
    DistanceInfo(
        canonical_name="dtw cost matrix",
        aka={"dtw cost matrix", "dynamic time warping cost matrix"},
        base_distance_class=_DtwDistanceCostMatrix,
        dist_func=_DtwDistanceCostMatrix().distance,
    ),
]
