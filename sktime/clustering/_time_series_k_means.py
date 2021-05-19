# -*- coding: utf-8 -*-
import numpy as np
from typing import List
import pandas as pd

from sktime.clustering.base.base_types import (
    Metric_Parameter,
    Metric_Function_Dict,
    Data_Frame,
    Series,
    Numpy_Array,
)
from sktime.clustering.base.base import ClusterMixin, Init_Algo, Init_Algo_Dict
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
from sktime.clustering.base.base import CenterInitializerMixin
from sktime.clustering._center_initializers import (
    RandomCenterInitializer,
    KMeansPlusPlusInitializer,
)
from sktime.utils.data_processing import from_nested_to_2d_array


class TimeSeriesKMeans(ClusterMixin):
    """
    TODO:
    Algorithm specific:
    Need to implement alternative averaging methods i.e. (barycenter
    averaging) for dtw.

    Need to implement more init algorithms i.e. (K means ++)

    Probably want to rerun the algorithm multiple times and take the
    best result as the init can be so impactful on final result

    Improve efficiency of distance calculation either want to precompute
    or ideally use a pairwise with the sktime distances (I dont know
    if this is already a thing in sktime)

    Multivariate support. While I've coded this with the intention to
    support it I doubt it will work with multivariate so need to code
    and improve that

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

    __init_algorithms: Init_Algo_Dict = {
        "random": RandomCenterInitializer,
        "k_means_plus_plus": KMeansPlusPlusInitializer,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        init_algorithm: Init_Algo = "random",
        max_iter: int = 300,
        verbose: bool = False,
        metric: Metric_Parameter = "dtw",
    ):
        """

        TODO:
        Add kmeans++ initilisation and potentially some other
        initialisations
        Add multivariate support
        Optimisation

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

        """
        if isinstance(init_algorithm, str):
            init_algorithm = TimeSeriesKMeans.__init_algorithms[init_algorithm]

        if isinstance(metric, str):
            metric = TimeSeriesKMeans.__metric_dict[metric]

        self.n_clusters: int = n_clusters
        self.n_init: int = n_init
        self.init_algorithm: CenterInitializerMixin = init_algorithm
        self.max_iter: int = max_iter
        self.verbose: bool = verbose
        self.metric = metric

        self.__centers: Data_Frame = None

    def fit(self, X: Data_Frame) -> None:
        self.__centers = self.init_algorithm.initialize_centers(X, self.n_clusters)
        if self.verbose:
            print("Initialization complete")

        for _ in range(self.max_iter):
            self.__update_centers(X)

        if self.verbose:
            print("Created centers")

    def predict(self, X: Data_Frame) -> Series:
        if self.__centers is None:
            raise Exception("Fit must be run before predict")

        data = from_nested_to_2d_array(X, return_numpy=True)
        return self.__cluster_data(data)

    def get_centers(self):
        return self.__centers

    def __cluster_data(self, X: Numpy_Array) -> List[List[int]]:
        clusters_index = [[] for _ in range(len(self.__centers))]

        centers = from_nested_to_2d_array(self.__centers, return_numpy=True)

        for i in range(len(X)):
            series = X[i]
            curr_min = None
            curr_min_index = 0

            for j in range(self.n_clusters):
                center = centers[j]
                curr_dist = self.metric(np.array([center]), np.array([series]))

                if curr_min is None:
                    curr_min = curr_dist
                elif curr_dist < curr_min:
                    curr_min = curr_dist
                    curr_min_index = j

            clusters_index[curr_min_index].append(i)

        return clusters_index

    def __update_centers(self, X: Data_Frame):
        data = from_nested_to_2d_array(X, return_numpy=True)
        cluster_indexes = self.__cluster_data(data)

        new_centers = []
        for i in range(len(cluster_indexes)):
            cluster_index = cluster_indexes[i]
            values = from_nested_to_2d_array(X.iloc[cluster_index], return_numpy=True)
            average = [pd.Series(values.mean(axis=0))]
            # NOTE: For DTW you have to do barycenter averaging
            # the below is temporary while I figure out how to
            # implement that
            new_centers.append(average)

        new_centers = pd.DataFrame(new_centers)
        new_centers.columns = self.__centers.columns

        self.__centers = new_centers
