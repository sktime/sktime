# -*- coding: utf-8 -*-

"""Approximations for time series clusterers"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["Medoids"]

import numpy as np

from sktime.clustering.base._typing import NumpyArray, MetricFunction
from sktime.clustering.base import BaseApproximate
from sktime.clustering.base.clustering_utils import compute_pairwise_distances


class Medoids(BaseApproximate):
    """Medoids Approximate

    Parameters
    ----------
    series: Numpy_Array
        series to perform approximation on

    """

    def __init__(self, series: NumpyArray, metric: MetricFunction):
        super(Medoids, self).__init__(series)
        self.metric = metric

    def approximate(self) -> int:
        """
        Method called to get the approximation

        Returns
        -------
        int
            Index position of the approximation in the series
        """
        distance_matrix = compute_pairwise_distances(X=self.series, metric=self.metric)
        return np.argmin(sum(distance_matrix))
