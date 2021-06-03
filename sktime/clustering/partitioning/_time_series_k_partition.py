# -*- coding: utf-8 -*-
from typing import List
import pandas as pd

from sklearn.metrics.pairwise import (
    pairwise_distances_argmin_min,
)
from sktime.clustering.base.base_types import (
    Metric_Parameter,
    Metric_Function_Dict,
    Data_Frame,
    Series,
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
from sktime.distances.mpdist import mpdist
from sktime.clustering.base.base import BaseClusterCenterInitializer
from sktime.clustering.partitioning._center_initializers import (
    RandomCenterInitializer,
)
from sktime.utils.data_processing import from_nested_to_2d_array
from sktime.clustering.utils._utils import compute_pairwise_distances
from sktime.distances.elastic import euclidean_distance

__author__ = "Christopher Holder"


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
        "mpdist": mpdist,
    }

    # "k_means_plus_plus": KMeansPlusPlusCenterInitializer,
    __init_algorithms: Init_Algo_Dict = {
        "random": RandomCenterInitializer,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Init_Algo = "random",
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

            init_algorithm: str, default = random
                Algorithm that is used to initialise the cluster
                centers. See clustering/base/base_types.py
                for description of type. The following are the
                str supported types:
                'random' = random initialisation

            max_iter: int, default = 300
                Maximum number of iterations of time series k means
                for a single run.

            verbose: bool, default = False
                Verbosity mode.

            metric: Metric_Parameter, default = None
                The distance metric that is used to calculate the
                distance between points. See clustering/base/base_types.py
                for description. The following are the str supported types:
                'eucli' = euclidean distance,
                'dtw' = DTW distance
        """
        if isinstance(init_algorithm, str):
            init_algorithm = TimeSeriesKPartition.__init_algorithms[init_algorithm]

        if isinstance(metric, str):
            metric = TimeSeriesKPartition.__metric_dict[metric]

        self.n_clusters: int = n_clusters
        self.init_algorithm: BaseClusterCenterInitializer = init_algorithm
        self.max_iter: int = max_iter
        self.verbose: bool = verbose
        self.metric = metric

        self.__centers: Data_Frame = None

    def fit(self, X: Data_Frame) -> None:
        """
        Method that is used to fit the time seires k
        partition model on dataset X

        Parameters
        ----------
        X: Data_Frame
            sktime data_frame to train the model on
        """
        center_algo: BaseClusterCenterInitializer = self.init_algorithm(
            X, self.n_clusters
        )
        self.__centers = center_algo.initialize_centers()

        for _ in range(self.max_iter):
            self.__update_centers(X)

    def predict(self, X: Data_Frame) -> Series:
        """
        Method used to perform a prediction from the trained
        time series k partition clustering algorithm

        Parameters
        ----------
        X: Data_Frame
            sktime data_frame to predict clusters for

        Returns
        -------
        List[List[int]]
            2d array, each sub list contains the indexes that
            belong to that cluster
        """
        if self.__centers is None:
            raise Exception("Fit must be run before predict")

        data = from_nested_to_2d_array(X, return_numpy=True)
        return self.__cluster_data(data)

    def get_centers(self) -> Data_Frame:
        """
        Method used to get the centers of the clustering
        algorithm

        Returns
        -------
        Data_Frame
            sktime data_frame containing the centers of the
            clusters
        """
        return self.__centers

    def calculate_new_centers(self, cluster_values: Numpy_Array) -> Numpy_Array:
        """
        Method to be implemented by parent defining how centers
        are calculated based on each iteration of k_partition

        Parameters
        ----------
        X: Numpy_Array
            Array of values that are part of the same cluster
            to calculate new centers from
        """
        raise NotImplementedError("abstract method")

    def __cluster_data(self, X: Numpy_Array) -> List[List[int]]:
        clusters_index = [[] for _ in range(len(self.__centers))]

        centers = from_nested_to_2d_array(self.__centers, return_numpy=True)

        for i in range(len(X)):
            pairwise_min = compute_pairwise_distances(
                metric=self.metric,
                X=[X[i]],
                Y=centers,
                pairwise_func=pairwise_distances_argmin_min,
            )
            clusters_index[pairwise_min[0][0]].append(i)

        return clusters_index

    def __update_centers(self, X: Data_Frame):
        data = from_nested_to_2d_array(X, return_numpy=True)
        cluster_indexes = self.__cluster_data(data)

        new_centers = []
        for i in range(len(cluster_indexes)):
            cluster_index = cluster_indexes[i]
            values = from_nested_to_2d_array(X.iloc[cluster_index], return_numpy=True)
            new_centers_i = self.calculate_new_centers(values)
            new_centers_i = [pd.Series(new_centers_i)]
            new_centers.append(new_centers_i)

        new_centers = pd.DataFrame(new_centers)
        new_centers.columns = self.__centers.columns

        self.__centers = new_centers
