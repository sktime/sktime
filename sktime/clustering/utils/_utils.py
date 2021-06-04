# -*- coding: utf-8 -*-
from typing import Callable
import numpy as np

from sktime.clustering.base.base_types import Numpy_Array, Metric_Function
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
    metric: Callable
        distance metric used to measure the distances
        between points

    X: Numpy_Array
        series that contains the values to take the
        pairwise of

    Y: Numpy_Array, defaults = None
        Other series to compute pairwise from. If unspecified
        will use the X series as Y

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
