# -*- coding: utf-8 -*-
"""Time Series K-Partition."""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["TimeSeriesKPartition"]

from typing import List
import numpy as np

from sklearn.metrics.pairwise import (
    pairwise_distances_argmin_min,
)
from sktime.clustering.base.base_types import (
    Metric_Parameter,
    Metric_Function_Dict,
    Numpy_Array,
)
from sktime.clustering.base.base import (
    BaseCluster,
    ClusterMixin,
    Init_Algo,
    Init_Algo_Dict,
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
)
from sktime.clustering.utils._utils import compute_pairwise_distances
from sktime.distances.elastic import euclidean_distance


class TimeSeriesKPartition(BaseCluster, ClusterMixin):
    """
    TODO:
    Algorithm specific:

    Need to implement more init algorithms i.e. (K means ++)

    Probably want to rerun the algorithm multiple times and take the
    best result as the init can be so impactful on final result

    Multivariate support. While I've coded this with the intention to
    support it I doubt it will work with multivariate so need to code
    and improve that
    """

    __metric_dict: Metric_Function_Dict = {
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
    __init_algorithms: Init_Algo_Dict = {
        "forgy": ForgyCenterInitializer,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Init_Algo = "forgy",
        max_iter: int = 300,
        verbose: bool = False,
        metric: Metric_Parameter = "dtw",
    ):
        """
        Constructor for time_series_k_partition clusterer

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

        """
        super(TimeSeriesKPartition, self).__init__()

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.metric = metric
        self.init_algorithm = init_algorithm

        self.__centers: Numpy_Array = None
        self._init_algorithm = None
        self._metric = None

    def _fit(self, X: Numpy_Array) -> None:
        """
        Method that is used to fit the time series k
        partition model on dataset X

        Parameters
        ----------
        X: Numpy_Array
            Numpy array of series to train the model on

        Returns
        -------
        self
            Fitted estimator
        """
        center_algo: BaseClusterCenterInitializer = self._init_algorithm(
            X, self.n_clusters
        )
        self.__centers = center_algo.initialize_centers()

        for _ in range(self.max_iter):
            self.__update_centers(X)

    def _predict(self, X: Numpy_Array) -> Numpy_Array:
        """
        Method used to perform a prediction from the trained
        time series k partition clustering algorithm

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

        if self.__centers is None:
            raise Exception("Fit must be run before predict")

        return self.__cluster_data(X)

    def get_centers(self) -> Numpy_Array:
        """
        Method used to get the centers of the clustering
        algorithm

        Returns
        -------
        Numpy_Array
            Containing the values of the centers
        """
        return self.__centers

    def calculate_new_centers(self, cluster_values: Numpy_Array) -> Numpy_Array:
        """
        Method to be implemented by parent defining how centers
        are calculated based on each iteration of k_partition

        Parameters
        ----------
        cluster_values: Numpy_Array
            Array of values that are part of the same cluster
            to calculate new centers from
        """
        raise NotImplementedError("abstract method")

    def _check_params(self, X: Numpy_Array):
        """
        Method used to check the parameters passed

        Parameters
        ----------
        X: Numpy_Array
            Dataset to be validate parameters against
        """
        if isinstance(self.init_algorithm, str):
            self._init_algorithm = TimeSeriesKPartition.__init_algorithms[
                self.init_algorithm
            ]

        if isinstance(self.metric, str):
            self._metric = TimeSeriesKPartition.__metric_dict[self.metric]

        super(TimeSeriesKPartition, self)._check_params(X)

    def __cluster_data(self, X: Numpy_Array) -> List[List[int]]:
        cluster_indexes = []

        for i in range(len(X)):
            pairwise_min = compute_pairwise_distances(
                metric=self._metric,
                X=[X[i]],
                Y=self.__centers,
                pairwise_func=pairwise_distances_argmin_min,
            )
            index = pairwise_min[0][0]
            cluster_indexes.append(index)

        return cluster_indexes

    def __update_centers(self, data: Numpy_Array):
        cluster_indexes = self.__cluster_data(data)

        cluster_values = TimeSeriesKPartition.get_cluster_values(
            cluster_indexes, data, self.n_clusters
        )

        new_centers = []
        for cluster_series in cluster_values:
            new_centers_i = self.calculate_new_centers(cluster_series)
            new_centers.append(new_centers_i)

        self.__centers = np.array(new_centers)

    @staticmethod
    def get_cluster_values(cluster_indexes: Numpy_Array, data: Numpy_Array, k: int):
        cluster_values = [[] for _ in range(k)]

        for i in range(len(cluster_indexes)):
            index = cluster_indexes[i]
            cluster_values[index].append(data[i])

        values = []
        for arr in cluster_values:
            values.append(np.array(arr))
        return np.array(values, dtype=object)
