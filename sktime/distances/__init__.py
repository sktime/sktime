# -*- coding: utf-8 -*-
"""Distance computation."""

__all__ = [
    "distance",
    "distance_factory",
    "pairwise_distance",
    "euclidean_distance",
    "squared_distance",
    "dtw_distance",
    "ddtw_distance",
    "wdtw_distance",
    "wddtw_distance",
    "edr_distance",
    "erp_distance",
    "lcss_distance",
    "msm_distance",
    "LowerBounding",
    "twe_distance",
]

from sktime.distances._distance import (
    ddtw_distance,
    distance,
    distance_factory,
    dtw_distance,
    edr_distance,
    erp_distance,
    euclidean_distance,
    lcss_distance,
    pairwise_distance,
    squared_distance,
    wddtw_distance,
    wdtw_distance,
)

# todo: replace these or remove these (placeholders for C removal)
from sktime.distances.elastic import msm_distance
from sktime.distances.elastic import msm_distance as twe_distance
from sktime.distances.lower_bounding import LowerBounding
