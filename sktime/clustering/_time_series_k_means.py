# -*- coding: utf-8 -*-
"""Time Series K-Means Clusterer."""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["TimeSeriesKMeans"]

from typing import List
from sktime.clustering.base.base_types import (
    Metric_Parameter,
    Data_Frame,
    Numpy_Array,
)
from sktime.clustering.base.base import (
    BaseCluster,
    Init_Algo,
    Averaging_Algo,
    Averaging_Algo_Dict,
)
from sktime.clustering.partitioning._averaging_metrics import (
    BarycenterAveraging,
    MeanAveraging,
)
from sktime.clustering.partitioning._time_series_k_partition import TimeSeriesKPartition


class TimeSeriesKMeans(TimeSeriesKPartition, BaseCluster):
    """Time Series K-Means Clusterer.

    This is a work in progress.
    """

    __averaging_algorithm_dict: Averaging_Algo_Dict = {
        "dba": BarycenterAveraging,
        "mean": MeanAveraging,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Init_Algo = "random",
        max_iter: int = 300,
        verbose: bool = False,
        metric: Metric_Parameter = "dtw",
        averaging_algorithm: Averaging_Algo = "mean",
        averaging_algorithm_iterations: int = 50,
    ):
        """Constructor for TimeSeiresKMeans clusterer

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
                distance between points.

            averaging_algorithm: Averaging_Algo
                The method used to create the average from a cluster

            averaging_algorithm_iterations: int
                Where appropriate (i.e. DBA) the average is refined by
                iterations. This is the number of times it is refined
        """
        super(TimeSeriesKMeans, self).__init__(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            max_iter=max_iter,
            verbose=verbose,
            metric=metric,
        )
        metric_str = None
        if isinstance(metric, str):
            metric_str = metric

        if isinstance(averaging_algorithm, str):
            if metric_str is not None and averaging_algorithm == "auto":
                if metric_str == "dtw":
                    averaging_algorithm = "dba"
                else:
                    averaging_algorithm = "mean"

            self.averaging_algorithm = TimeSeriesKMeans.__averaging_algorithm_dict[
                averaging_algorithm
            ]
        else:
            self.averaging_algorithm = averaging_algorithm
        self.averaging_algorithm_iterations = averaging_algorithm_iterations

    def fit(self, X: Data_Frame) -> None:
        """
        Method that is used to fit the time series k
        means model on dataset X

        Parameters
        ----------
        X: Data_Frame
            sktime data_frame to train the model on
        """
        super(TimeSeriesKMeans, self).fit(X)

    def predict(self, X: Data_Frame) -> List[List[int]]:
        """
        Method used to perform a prediction from the trained
        time series k means

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
        return super(TimeSeriesKMeans, self).predict(X)

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
        average_algorithm = self.averaging_algorithm(
            cluster_values, self.averaging_algorithm_iterations
        )
        return average_algorithm.average()
