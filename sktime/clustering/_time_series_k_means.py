# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Tuple

from sktime.clustering.base.base import ClusterMixin
from sktime.clustering.base.cluster_types import (
    Metric_Parameter,
    Init_Algo,
    Metric_Function_Dict,
    Init_Algo_Dict,
    Data_Frame,
)

from sktime.distances.elastic_cython import (
    ddtw_distance,
    dtw_distance,
    erp_distance,
    lcss_distance,
    msm_distance,
    twe_distance,
    wddtw_distance,
    wdtw_distance
)
from sktime.distances.mpdist import mpdist


def random_center_initializer(X, n_clusters) -> Tuple[List, List]:
    """
    Generates k number of centers by randomly selecting
    values from X

    TODO:
    This assumes that this is a univariate TS need
    to add multivariate support

    Parameters
    ----------
    X: sktime_df
        Training instances to cluster

    n_clusters: int
        Number of clusters to create

    Returns:
    --------
    centroids:
        Contains the centroid values randomly selected

    centroid_indexes:
        Contains the indexes of the randomly selected
        centroids

    """
    centroids = []
    centroids_indexes = []
    n = np.shape(X)[0]

    def create_center():
        rand_index = np.random.randint(0, n - 1)
        if rand_index in centroids_indexes:
            return create_center()
        centroids.append(X["dim_0"][rand_index])
        centroids_indexes.append(rand_index)

    [create_center() for _ in range(n_clusters)]
    return centroids, centroids_indexes


class TimeSeiresKMeans(ClusterMixin):

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

    __init_algorithms: Init_Algo_Dict = {
        "random": random_center_initializer
    }

    def __init__(self,
                 n_clusters: int = 8,
                 n_init: int = 10,
                 init_algorithm: Init_Algo = "random",
                 max_iter: int = 300,
                 verbose: bool = False,
                 metric: Metric_Parameter = None):
        """

        TODO:
        Add kmeans++ initilisation and potentially some other
        initialisations
        Add multivariate support

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
                centers. See clustering/base/cluster_types.py
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
                distance between points. See clustering/base/cluster_types.py
                for description. The following are the str supported types:
                'eucli' = euclidean distance,
                'dtw' = DTW distance

        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.init_algorithm = init_algorithm
        self.max_iter = max_iter
        self.verbose = verbose
        if isinstance(metric, str):
            self.init_algorithm = TimeSeiresKMeans.__metric_dict[metric]
        else:
            self.init_algorithm = metric
        self.metric = metric

        self.__clusters = []
        self.__centers = []

    def fit(self, X: Data_Frame):
        # print("\n================")
        # print("Running fit")

        # Create the initial centers
        self.__create_initial_centers(X)
        centroids, centroids_index = self.init_algorithm(X)
        # print(centroids_index)

        self.__update_clusters(X)
        # print("Done fit")
        # while iteration < self.max_iter:
        #     iteration += 1
        #     print("here")

    def predict(self, X: Data_Frame):
        pass

    def __create_initial_centers(self, X: Data_Frame):
        pass

    def __kmeans_plus_plus_initializer(self):
        pass
