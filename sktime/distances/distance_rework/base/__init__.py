# -*- coding: utf-8 -*-
"""Base class and type for numba distances."""
__author__ = ["chrisholder", "TonyBagnall"]
__all__ = [
    "BaseDistance",
    "DistanceCostCallable",
    "DistanceFactoryCallable",
    "DistancePairwiseCallable",
    "ValidCallableTypes",
    "MetricInfo",
    "DistanceAlignmentPathCallable",
    "AlignmentPathReturn",
]

from sktime.distances.distance_rework.base._base import BaseDistance, MetricInfo
from sktime.distances.distance_rework.base._types import (
    AlignmentPathReturn,
    DistanceAlignmentPathCallable,
    DistanceCostCallable,
    DistanceFactoryCallable,
    DistancePairwiseCallable,
    ValidCallableTypes,
)
