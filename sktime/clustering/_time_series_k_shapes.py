# -*- coding: utf-8 -*-
"""Time series K-shapes clusterer"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["TimeSeriesKShapes"]

from sktime.clustering.base.base_types import (
    Numpy_Array,
)
from sktime.clustering.base.base import (
    BaseCluster,
)
from sktime.clustering.partitioning._time_series_k_partition import TimeSeriesKPartition


class TimeSeriesKShapes(TimeSeriesKPartition, BaseCluster):
    def calculate_new_centers(self, cluster_values: Numpy_Array) -> Numpy_Array:
        pass
