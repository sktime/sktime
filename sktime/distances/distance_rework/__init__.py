# -*- coding: utf-8 -*-
"""Distance module."""
__all__ = [
    "BaseDistance",
    "ElasticDistance",
    "DistanceCallable",
    "_EuclideanDistance",
    "_SquaredDistance",
    "_DtwDistance",
    "_WdtwDistance",
    "_DdtwDistance",
    "_WddtwDistance",
    "_EdrDistance",
    "LocalDistanceCallable",
    "_ErpDistance",
    "_LcssDistance",
    "_TweDistance",
    "_MsmDistance",
    "distance",
    "distance_factory",
    "distance_alignment_path_factory",
    "distance_alignment_path",
    "pairwise_distance",
]

from sktime.distances.distance_rework._base import (
    BaseDistance,
    DistanceCallable,
    LocalDistanceCallable,
    ElasticDistance,
)
from sktime.distances.distance_rework._ddtw import _DdtwDistance
from sktime.distances.distance_rework._dtw import _DtwDistance
from sktime.distances.distance_rework._edr import _EdrDistance
from sktime.distances.distance_rework._erp import _ErpDistance
from sktime.distances.distance_rework._euclidean import _EuclideanDistance
from sktime.distances.distance_rework._lcss import _LcssDistance
from sktime.distances.distance_rework._msm import _MsmDistance
from sktime.distances.distance_rework._squared import _SquaredDistance
from sktime.distances.distance_rework._twe import _TweDistance
from sktime.distances.distance_rework._wddtw import _WddtwDistance
from sktime.distances.distance_rework._wdtw import _WdtwDistance
from sktime.distances.distance_rework._distance import (
    distance,
    distance_factory,
    distance_alignment_path_factory,
    distance_alignment_path,
    pairwise_distance
)
