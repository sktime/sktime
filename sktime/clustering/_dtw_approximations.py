# -*- coding: utf-8 -*-
import numpy as np
from typing import Callable

from sktime.clustering.base.base_types import Numpy_Array, Metric_Function
from sktime.clustering.base.base import BaseApproximate
from sklearn.metrics.pairwise import pairwise_distances

__author__ = "Christopher Holder"


def compute_pairwise_distances(
    metric: Metric_Function,
    X: Numpy_Array,
    Y: Numpy_Array = None,
    pairwise_func: Callable = pairwise_distances,
):
    """
    Method used to compute a pairwise distance matrix
    for a series

    Parameters
    ----------
    series: Numpy_Array
        series that contains the values to take the
        pairwise of

    metric: Callable
        distance metric used to measure the distances
        between points

    pairwise_func: Callable, defaults = pairwise_distance
        pairwise function to execute

    Returns
    -------
    Results in the format the pairwise_func returns. If
    using the default then it will be a distance matrix

    """

    def dist_wrapper(first, second):
        if first.ndim <= 1:
            first = np.array([first])
        if second.ndim <= 1:
            second = np.array([second])
        return metric(first, second)

    if Y is None:
        Y = X
    return pairwise_func(X=X, Y=Y, metric=dist_wrapper)


class DTWMedoids(BaseApproximate):
    def __init__(self, series: Numpy_Array, metric: Metric_Function):
        super(DTWMedoids, self).__init__(series)
        self.metric = metric

    def approximate(self):
        distance_matrix = compute_pairwise_distances(X=self.series, metric=self.metric)
        return np.argmin(sum(distance_matrix))
