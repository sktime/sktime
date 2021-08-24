# -*- coding: utf-8 -*-
import numpy as np
from typing import List
from scipy.spatial.distance import cdist

from sktime.metrics.distances.base.base import BaseDistance, BasePairwise


class ScipyDistance(BaseDistance, BasePairwise):
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

    def __init__(self, metric: str, kwargs={}):
        self.metric = metric
        self.kwargs = kwargs
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
        return float(np.sum(cdist(x, y, metric=self.metric, **self.kwargs)))

    def _pairwise(self, x: np.ndarray, y: np.ndarray, symmetric: bool) -> np.ndarray:
        """
        Method to compute a pairwise distance on a matrix (i.e. distance between each
        ts in the matrix)

        Parameters
        ----------
        x: np.ndarray
            First matrix of multiple time series
        y: np.ndarray
            Second matrix of multiple time series.
        symmetric: bool
            boolean that is true when the two time series are equal to each other

        Returns
        -------
        np.ndarray
            Matrix containing the pairwise distance between each point
        """
        x_size = x.shape[0]
        y_size = y.shape[0]

        pairwise_matrix = np.zeros((x_size, y_size))

        for i in range(x_size):
            curr_x = x[i]
            for j in range(y_size):
                if symmetric and j < i:
                    pairwise_matrix[i, j] = pairwise_matrix[j, i]
                else:
                    pairwise_matrix[i, j] = self.distance(curr_x, y[j])

        return pairwise_matrix

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
