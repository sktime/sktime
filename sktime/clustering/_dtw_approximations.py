# -*- coding: utf-8 -*-
import numpy as np

from sktime.clustering.base.base_types import Numpy_Array, Metric_Function
from sktime.clustering.base.base import BaseApproximate
from sktime.clustering._utils import compute_pairwise_distances

__author__ = "Christopher Holder"


class DTWMedoids(BaseApproximate):
    def __init__(self, series: Numpy_Array, metric: Metric_Function):
        super(DTWMedoids, self).__init__(series)
        self.metric = metric

    def approximate(self):
        distance_matrix = compute_pairwise_distances(X=self.series, metric=self.metric)
        return np.argmin(sum(distance_matrix))
