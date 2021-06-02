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
    Averaging_Algo,
    Averaging_Algo_Dict,
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
from sktime.clustering._center_initializers import (
    RandomCenterInitializer,
)
from sktime.utils.data_processing import from_nested_to_2d_array
from sktime.clustering._averaging_metrics import BarycenterAveraging, MeanAveraging

__author__ = "Christopher Holder"


class TimeSeriesKMeans(BaseCluster, ClusterMixin):
    """
    TODO:
    Algorithm specific:

    Need to implement more init algorithms i.e. (K means ++)

    Probably want to rerun the algorithm multiple times and take the
    best result as the init can be so impactful on final result

    Multivariate support. While I've coded this with the intention to
    support it I doubt it will work with multivariate so need to code
    and improve that

    Fix the future warning from sklearn

    Validation/practices:
    Check params
    Add comments
    Unit tests
    Documentation
    """

    __metric_dict: Metric_Function_Dict = {
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

    __averaging_algorithm_dict: Averaging_Algo_Dict = {
        "dba": BarycenterAveraging,
        "mean": MeanAveraging,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        init_algorithm: Init_Algo = "random",
        max_iter: int = 300,
        verbose: bool = False,
        metric: Metric_Parameter = "dtw",
        averaging_algorithm: Averaging_Algo = "auto",
    ):
        """
        Constructor for time_series_k_means clusterer

        Parameters
        ----------
            n_clusters: int, default = 8
                The number of clusters to form as the number of
                centroids to generate.

            n_init: int, default = 10
                Number of time the k-means algorithm will be run
                with different centroid seeds. The final result
                will be the best output of n_init consecutive runs
                in terms of inertia.

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

            averaging_algorithm: Averaging_Algo
                The method used to create the average from a cluster

        """
        if isinstance(init_algorithm, str):
            init_algorithm = TimeSeriesKMeans.__init_algorithms[init_algorithm]

        metric_str = None
        if isinstance(metric, str):
            metric_str = metric
            metric = TimeSeriesKMeans.__metric_dict[metric]

        if isinstance(averaging_algorithm, str):
            if metric_str is not None and averaging_algorithm == "auto":
                if metric_str == "dtw":
                    averaging_algorithm = "dba"
                else:
                    averaging_algorithm = "mean"

            self.averaging_algorithm = TimeSeriesKMeans.__averaging_algorithm_dict[
                averaging_algorithm
            ]

        self.n_clusters: int = n_clusters
        self.n_init: int = n_init
        self.init_algorithm: BaseClusterCenterInitializer = init_algorithm
        self.max_iter: int = max_iter
        self.verbose: bool = verbose
        self.metric = metric

        self.__centers: Data_Frame = None

    def fit(self, X: Data_Frame) -> None:
        """
        Method that is used to fir the time seires k-means
        clustering algorithm on dataset X

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
        Method used to perfor a prediction from the trained
        time series k-means clustering algorithm

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

    def __cluster_data(self, X: Numpy_Array) -> List[List[int]]:
        clusters_index = [[] for _ in range(len(self.__centers))]

        centers = from_nested_to_2d_array(self.__centers, return_numpy=True)

        for i in range(len(X)):
            pairwise_min = pairwise_distances_argmin_min([X[i]], centers, self.metric)
            clusters_index[pairwise_min[0][0]].append(i)

        return clusters_index

    def __update_centers(self, X: Data_Frame):
        data = from_nested_to_2d_array(X, return_numpy=True)
        cluster_indexes = self.__cluster_data(data)

        new_centers = []
        for i in range(len(cluster_indexes)):
            cluster_index = cluster_indexes[i]
            values = from_nested_to_2d_array(X.iloc[cluster_index], return_numpy=True)
            average_algo: Averaging_Algo = self.averaging_algorithm(values)
            average = [pd.Series(average_algo.average())]
            new_centers.append(average)

        new_centers = pd.DataFrame(new_centers)
        new_centers.columns = self.__centers.columns

        self.__centers = new_centers
