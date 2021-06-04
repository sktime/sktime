# -*- coding: utf-8 -*-
from typing import List
from sktime.clustering.base.base_types import (
    Metric_Parameter,
    Data_Frame,
    Numpy_Array,
)
from sktime.clustering.base.base import (
    BaseCluster,
    Init_Algo,
)
from sktime.clustering.partitioning._time_series_k_partition import TimeSeriesKPartition
from sktime.clustering.partitioning._dtw_approximations import Medoids

__author__ = "Christopher Holder"


class TimeSeriesKMedoids(TimeSeriesKPartition, BaseCluster):
    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Init_Algo = "random",
        max_iter: int = 300,
        verbose: bool = False,
        metric: Metric_Parameter = "dtw",
    ):
        """
        Constructor for TimeSeiresKMedoids clusterer

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
                centers.

            max_iter: int, default = 300
                Maximum number of iterations of time series k means
                for a single run.

            verbose: bool, default = False
                Verbosity mode.

            metric: Metric_Parameter, default = None
                The distance metric that is used to calculate the
                distance between points. See clustering/base/base_types.py
                for description. The following are the str supported types:
        """
        super(TimeSeriesKMedoids, self).__init__(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            max_iter=max_iter,
            verbose=verbose,
            metric=metric,
        )

    def fit(self, X: Data_Frame) -> None:
        """
        Method that is used to fit the time seires k
        medoids model on dataset X

        Parameters
        ----------
        X: Data_Frame
            sktime data_frame to train the model on
        """
        super(TimeSeriesKMedoids, self).fit(X)

    def predict(self, X: Data_Frame) -> List[List[int]]:
        """
        Method used to perform a prediction from the trained
        time series k medoids

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
        return super(TimeSeriesKMedoids, self).predict(X)

    def calculate_new_centers(self, cluster_values: Numpy_Array) -> Numpy_Array:
        """
        Method used to define how the centers are calculated

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
        medoid = Medoids(cluster_values, self.metric)
        medoid_index = medoid.approximate()
        return cluster_values[medoid_index]
