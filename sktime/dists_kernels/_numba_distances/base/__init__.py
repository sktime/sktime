"""Base class and type for numba distances."""

__author__ = ["chrisholder", "TonyBagnall"]
__all__ = [
    "NumbaDistance",
    "DistanceCallable",
    "DistanceFactoryCallable",
    "DistancePairwiseCallable",
    "ValidCallableTypes",
    "MetricInfo",
    "DistanceAlignmentPathCallable",
    "AlignmentPathReturn",
]

from sktime.dists_kernels._numba_distances.base._base import MetricInfo, NumbaDistance
from sktime.dists_kernels._numba_distances.base._types import (
    AlignmentPathReturn,
    DistanceAlignmentPathCallable,
    DistanceCallable,
    DistanceFactoryCallable,
    DistancePairwiseCallable,
    ValidCallableTypes,
)
