# -*- coding: utf-8 -*-
"""Time series K-medoids clusterer"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["TimeSeriesKMedoids"]

from sktime.clustering.base._typing import (
    MetricParameter,
    NumpyArray,
    NumpyOrDF,
    InitAlgo,
    NumpyRandomState,
)
from sktime.clustering.partitioning._lloyds_partitioning import (
    TimeSeriesLloydsPartitioning,
)
from sktime.clustering.partitioning._cluster_approximations import Medoids
from sktime.clustering.base import BaseClusterer


class TimeSeriesKMedoids(TimeSeriesLloydsPartitioning):
    """Time Series K-Medoids Clusterer

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

    random_state: NumpyRandomState, default = np.random.RandomState(1)
        Generator used to initialise the centers.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: InitAlgo = "forgy",
        max_iter: int = 300,
        verbose: bool = False,
        metric: MetricParameter = "dtw",
        random_state: NumpyRandomState = None,
    ):
        super(TimeSeriesKMedoids, self).__init__(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            max_iter=max_iter,
            verbose=verbose,
            metric=metric,
            random_state=random_state,
        )

    def fit(self, X: NumpyOrDF, y: NumpyOrDF = None) -> BaseClusterer:
        """
        Method that is used to fit the clustering algorithm
        on the dataset X

        Parameters
        ----------
        X: Numpy array or Dataframe
            sktime data_frame or numpy array to train the model on

        y: Numpy array of Dataframe, default = None
            sktime data_frame or numpy array that is the labels for training.
            Unlikely to be used for clustering but kept for consistency

        Returns
        -------
        self
            Fitted estimator
        """

        return super(TimeSeriesKMedoids, self).fit(X)

    def predict(self, X: NumpyOrDF) -> NumpyArray:
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

    def calculate_new_centers(self, cluster_values: NumpyArray) -> NumpyArray:
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
        if self._metric is None:
            self._check_params(cluster_values)

        medoid = Medoids(cluster_values, self._metric)
        medoid_index = medoid.approximate()
        return cluster_values[medoid_index]
