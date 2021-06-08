# -*- coding: utf-8 -*-

"""Approximations for time series clusterers"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["Medoids"]

import numpy as np

from sktime.clustering.base.base_types import Numpy_Array, Metric_Function
from sktime.clustering.base.base import BaseApproximate
from sktime.clustering.utils._utils import compute_pairwise_distances


class Medoids(BaseApproximate):
    def __init__(self, series: Numpy_Array, metric: Metric_Function):
        super(Medoids, self).__init__(series)
        self.metric = metric

    def approximate(self) -> int:
        distance_matrix = compute_pairwise_distances(X=self.series, metric=self.metric)
        return np.argmin(sum(distance_matrix))
