# -*- coding: utf-8 -*-
"""Time series K-medoids clusterer"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["TimeSeriesKMedoids"]

from sktime.clustering.base.base_types import Metric_Parameter, Numpy_Array, Numpy_Or_DF
from sktime.clustering.base.base import (
    Init_Algo,
)
from sktime.clustering.partitioning._time_series_k_partition import TimeSeriesKPartition
from sktime.clustering.partitioning._cluster_approximations import Medoids


class TimeSeriesKMedoids(TimeSeriesKPartition):
    """Time Series K-Medoids Clusterer.

    This is a work in progress
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Init_Algo = "forgy",
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

            init_algorithm: Init_Algo or str, default = forgy
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
        super(TimeSeriesKMedoids, self).__init__(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            max_iter=max_iter,
            verbose=verbose,
            metric=metric,
        )

    def fit(self, X: Numpy_Or_DF) -> None:
        """
        Method that is used to fit the clustering algorithm
        on the dataset X

        Parameters
        ----------
        X: Numpy array or Dataframe
            sktime data_frame or numpy array to train the model on

        Returns
        -------
        self
            Fitted estimator
        """

        return super(TimeSeriesKMedoids, self).fit(X)

    def predict(self, X: Numpy_Or_DF) -> Numpy_Array:
        """
        Method used to perform a prediction from the already
        trained clustering algorithm

        Parameters
        ----------
        X: Numpy array or Dataframe
            sktime data_frame or numpy array to predict
            cluster for

        Returns
        -------
        Numpy_Array: np.array
            Index of the cluster each sample belongs to
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
        medoid = Medoids(cluster_values, self._metric)
        medoid_index = medoid.approximate()
        return cluster_values[medoid_index]
