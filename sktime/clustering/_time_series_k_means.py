# -*- coding: utf-8 -*-
import numpy
import numpy as np
from typing import List, Tuple

import pandas as pd

from sktime.clustering.base.base import ClusterMixin
from sktime.clustering.base.cluster_types import (
    Metric_Parameter,
    Init_Algo,
    Metric_Function_Dict,
    Init_Algo_Dict,
    Data_Frame,
    Series
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
from sktime.utils.data_processing import from_nested_to_2d_array


def random_center_initializer(data: Data_Frame, n_clusters) -> Tuple[List[Series], List[int]]:
    """
    Generates k number of centers by randomly selecting
    values from X

    TODO:
    This assumes that this is a univariate TS need
    to add multivariate support

    Parameters
    ----------
    data: sktime_df
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
    n = np.shape(data)[0]

    def create_center():
        rand_index = np.random.randint(0, n - 1)
        if rand_index in centroids_indexes:
            return create_center()
        centroids.append(data["dim_0"][rand_index])
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
                 metric: Metric_Parameter = "dtw"):
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

        if isinstance(init_algorithm, str):
            self.init_algorithm = TimeSeiresKMeans.__init_algorithms[init_algorithm]
        else:
            self.init_algorithm = init_algorithm

        if isinstance(metric, str):
            self.metric = TimeSeiresKMeans.__metric_dict[metric]
        else:
            self.metric = metric

        self.__centers: List[Series] = []

    def fit(self, X: Data_Frame):
        print("Running fit")

        data = X.copy(deep=True)

        # Create the initial centers
        self.__centers, centroids_index = self.init_algorithm(data, self.n_clusters)

        for i in range(self.max_iter):
            self.__update_clusters(data)

        i = 0
        for center in self.__centers:
            print("Center {}: {}".format(i, center[0]))
            i += 1

        print("Done fit")

    def predict(self, X: Data_Frame):
        print("Running predict")
        if not self.__centers:
            raise Exception("Fit must be run before predict")

        clusters = self.__cluster(X)
        converted_clusters = []
        for cluster in clusters:
            temp = []
            for series in cluster:
                temp.append(pd.Series(series[0]))
            converted_clusters.append(temp)

        print(len(clusters[0]))
        print(len(clusters[1]))
        print(len(clusters[2]))
        return converted_clusters

    def get_centroids(self):
        return self.__centers


    def __update_clusters(self, data: Data_Frame):

        clusters = self.__cluster(data)

        new_centers: List[Series] = []

        for cluster in clusters:
            cluster = np.array(cluster)
            average = cluster.mean(axis=0)
            new_centers.append(pd.Series(average[0]))

        self.__centers = new_centers

    def __cluster(self, data: Data_Frame):

        clusters = [[] for _ in range(len(self.__centers))]

        numpy_data: np.array = (from_nested_to_2d_array(data)).to_numpy()
        numpy_data = [np.asarray([series])
                      for series in numpy_data]

        numpy_centers: np.array = [np.asarray([center.to_numpy()])
                                   for center in self.__centers]

        for series in numpy_data:
            curr_min = None
            curr_min_index = 0

            for i in range(len(numpy_centers)):
                center = numpy_centers[i]
                curr_dist = self.metric(center, series)

                if curr_min is None:
                    curr_min = curr_dist
                elif curr_dist < curr_min:
                    curr_min = curr_dist
                    curr_min_index = i

            clusters[curr_min_index].append(series)

        return clusters
