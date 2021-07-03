# -*- coding: utf-8 -*-
"""Time Series K-Partition."""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["TimeSeriesLloydsPartitioning"]

from typing import List
import numpy as np

from sklearn.metrics.pairwise import (
    pairwise_distances_argmin_min,
)
from sklearn.utils import check_random_state
from sktime.clustering.base._typing import (
    MetricParameter,
    MetricFunctionDict,
    NumpyArray,
    InitAlgo,
    InitAlgoDict,
    NumpyRandomState,
)
from sktime.clustering.base import (
    BaseClusterer,
)
from sktime.distances.elastic_cython import (
    ddtw_distance,
    dtw_distance,
    erp_distance,
    lcss_distance,
    msm_distance,
    twe_distance,
    wddtw_distance,
    wdtw_distance,
)
from sktime.clustering.base.base import BaseClusterCenterInitializer
from sktime.clustering.partitioning._center_initializers import (
    ForgyCenterInitializer,
    RandomCenterInitializer,
)
from sktime.clustering.base.clustering_utils import compute_pairwise_distances
from sktime.distances.elastic import euclidean_distance


class TimeSeriesLloydsPartitioning(BaseClusterer):
    """Time Series Lloyds partitioning algorithm
    Parameters
    ----------
    n_clusters: int, default = 8
        The number of clusters to form as the number of
        centroids to generate.
    init_algorithm: Init_Algo, default = forgy
        Algorithm that is used to initialise the cluster
        centers. str options are "forgy", "random" or
        "k-means++". If using custom center init algorithm
        then must be of type Init_Algo
    max_iter: int, default = 300
        Maximum number of iterations of time series k means
        for a single run.
    verbose: bool, default = False
        Verbosity mode.
    metric: Metric_Parameter, default = None
        The distance metric that is used to calculate the
        distance between points.
    random_state: NumpyRandomState, default = np.random.RandomState(1)
        Generator used to initialise the centers.
    """

    _metric_dict: MetricFunctionDict = {
        "euclidean": euclidean_distance,
        "dtw": dtw_distance,
        "ddtw": ddtw_distance,
        "wdtw": wdtw_distance,
        "wddtw": wddtw_distance,
        "lcss": lcss_distance,
        "erp": erp_distance,
        "msm": msm_distance,
        "twe": twe_distance,
    }

    # "k_means_plus_plus": KMeansPlusPlusCenterInitializer,
    _init_algorithms: InitAlgoDict = {
        "forgy": ForgyCenterInitializer,
        "random": RandomCenterInitializer,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: InitAlgo = "forgy",
        max_iter: int = 300,
        verbose: bool = False,
        metric: MetricParameter = "dtw",
        random_state: NumpyRandomState = None,
    ):
        super(TimeSeriesLloydsPartitioning, self).__init__()

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.metric = metric
        self.init_algorithm = init_algorithm
        self.random_state = random_state

        self._centers: NumpyArray = None
        self._init_algorithm = None
        self._metric = None
        self._random_state = None

    def _fit(self, X: NumpyArray, y: NumpyArray = None) -> BaseClusterer:
        """
        Method that contains the core logic to fit a cluster
        to training data
        Parameters
        ----------
        X: Numpy array
            Numpy array to train the model on
        y: Numpy array, default = None
            Numpy array that is the labels for training.
            Unlikely to be used for clustering but kept for consistency
        Returns
        -------
        self
            Fitted estimator
        """
        self._random_state = check_random_state(self.random_state)
        center_algo: BaseClusterCenterInitializer = self._init_algorithm(
            X, self.n_clusters, self.calculate_new_centers, self._random_state
        )
        self._centers = center_algo.initialize_centers()

        for _ in range(self.max_iter):
            self._update_centers(X)

    def _predict(self, X: NumpyArray) -> NumpyArray:
        """
        Method used to perform a prediction from the trained
        model
        Parameters
        ----------
        X: Numpy_Array
            Numpy_Array containing the time series to
            predict clusters for
        Returns
        -------
        Numpy_Array
            Index of the cluster each sample belongs to
        """
        return self._cluster_data(X)

    def get_centers(self) -> NumpyArray:
        """
        Method used to get the centers of the clustering
        algorithm
        Returns
        -------
        Numpy_Array
            Containing the values of the centers
        """
        return self._centers

    def calculate_new_centers(self, cluster_values: NumpyArray) -> NumpyArray:
        """
        Method to be implemented by parent defining how centers
        are calculated based on each iteration of k_partition
        Parameters
        ----------
        cluster_values: Numpy_Array
            Values to derive a center from (values in a cluster)
        Returns
        -------
        Numpy_Array
            Single value that is determined to be the center of
            the series
        """
        raise NotImplementedError("abstract method")

    def _check_params(self, X: NumpyArray):
        """
        Method used to check the parameters passed
        Parameters
        ----------
        X: Numpy_Array
            Dataset to be validate parameters against
        """
        if isinstance(self.init_algorithm, str):
            self._init_algorithm = TimeSeriesLloydsPartitioning._init_algorithms[
                self.init_algorithm
            ]

        if isinstance(self.metric, str):
            self._metric = TimeSeriesLloydsPartitioning._metric_dict[self.metric]

        super(TimeSeriesLloydsPartitioning, self)._check_params(X)

    def _cluster_data(self, X: NumpyArray) -> List[List[int]]:
        cluster_indexes = []

        for i in range(len(X)):
            pairwise_min = compute_pairwise_distances(
                metric=self._metric,
                X=[X[i]],
                Y=self._centers,
                pairwise_func=pairwise_distances_argmin_min,
            )
            index = pairwise_min[0][0]
            cluster_indexes.append(index)

        return cluster_indexes

    def _update_centers(self, data: NumpyArray):
        cluster_indexes = self._cluster_data(data)

        cluster_values = TimeSeriesLloydsPartitioning.get_cluster_values(
            cluster_indexes, data, self.n_clusters
        )

        new_centers = []
        for cluster_series in cluster_values:
            new_centers_i = self.calculate_new_centers(cluster_series)
            new_centers.append(new_centers_i)

        self._centers = np.array(new_centers)

    @staticmethod
    def get_cluster_values(cluster_indexes: NumpyArray, data: NumpyArray, k: int):
        cluster_values = [[] for _ in range(k)]

        for i in range(len(cluster_indexes)):
            index = cluster_indexes[i]
            cluster_values[index].append(data[i])

        values = []
        for arr in cluster_values:
            values.append(np.array(arr))
        return np.array(values, dtype=object)
