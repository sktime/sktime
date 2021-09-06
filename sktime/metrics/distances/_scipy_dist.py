# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Any
from scipy.spatial.distance import cdist

from sktime.metrics.distances.base.base import BaseDistance


class ScipyDistance(BaseDistance):
    """
    Class that supports the scipy distance functions for time series

    Parameters
    ----------
    metric: str
        str that is the name of the distance metric to use from scipy. Any of the
        following are valid distance metrics:
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’,
        ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’,
        ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’,
        ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’,
        ‘yule’.
    """

    def __init__(self, metric: str, kwargs=None):
        self.metric: str = metric
        self.kwargs: Any = kwargs
        super(ScipyDistance, self).__init__("scipy", {"scipy distance"})

    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Method used to compute the distance between two ts series

        Parameters
        ----------
        x: np.ndarray
            First time series
        y: np.ndarray
            Second time series

        Returns
        -------
        float
            Distance between time series x and time series y
        """
        if self.kwargs is None:
            kwargs = {}
        distances = cdist(x, y, metric=self.metric, **kwargs)
        dist_sum = 0.0
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                dist_sum += distances[i, j]

        return dist_sum

    @staticmethod
    def supported_scipy_distances() -> List:
        """
        Method used to get a list of support distances that are included in scipy

        Returns
        -------
        List[str]
            List of str distances that are valid scipy distances
        """
        return [
            "braycurtis",
            "canberra",
            "chebyshev",
            "cityblock",
            "correlation",
            "cosine",
            "dice",
            "euclidean",
            "hamming",
            "jaccard",
            "jensenshannon",
            "kulsinski",
            "mahalanobis",
            "matching",
            "minkowski",
            "rogerstanimoto",
            "russellrao",
            "seuclidean",
            "sokalmichener",
            "sokalsneath",
            "sqeuclidean",
            "yule",
        ]
