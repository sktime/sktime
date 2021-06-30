# -*- coding: utf-8 -*-
"""Time Series K-Means Clusterer."""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["TimeSeriesKMeans"]

from sktime.clustering.base._typing import (
    InitAlgo,
    AveragingAlgo,
    AveragingAlgoDict,
    MetricParameter,
    NumpyArray,
    NumpyOrDF,
    NumpyRandomState,
)
from sktime.clustering.base import BaseClusterer
from sktime.clustering.partitioning._averaging_metrics import (
    BarycenterAveraging,
    MeanAveraging,
)
from sktime.clustering.partitioning._lloyds_partitioning import (
    TimeSeriesLloydsPartitioning,
)


class TimeSeriesKMeans(TimeSeriesLloydsPartitioning):
    """Time Series K-Means Clusterer

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

    averaging_algorithm: Averaging_Algo, default = "mean"
        The method used to create the average from a cluster.
        str options are "dba" dtw barycenter averaging and
        "means" for mean average.

    averaging_algorithm_iterations: int, default = 10
        Where appropriate (i.e. DBA) the average is refined by
        iterations. This is the number of times it is refined

    random_state: NumpyRandomState, default = np.random.RandomState(1)
        Generator used to initialise the centers.
    """

    _averaging_algorithm_dict: AveragingAlgoDict = {
        "dba": BarycenterAveraging,
        "mean": MeanAveraging,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: InitAlgo = "forgy",
        max_iter: int = 300,
        verbose: bool = False,
        metric: MetricParameter = "dtw",
        averaging_algorithm: AveragingAlgo = "mean",
        averaging_algorithm_iterations: int = 10,
        random_state: NumpyRandomState = None,
    ):
        super(TimeSeriesKMeans, self).__init__(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            max_iter=max_iter,
            verbose=verbose,
            metric=metric,
            random_state=random_state,
        )
        self.averaging_algorithm = averaging_algorithm
        self.averaging_algorithm_iterations = averaging_algorithm_iterations
        self._averaging_algorithm = None

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
        return super(TimeSeriesKMeans, self).fit(X)

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
        return super(TimeSeriesKMeans, self).predict(X)

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

        average_algorithm = self._averaging_algorithm(
            cluster_values, self.averaging_algorithm_iterations
        )
        return average_algorithm.average()

    def _check_params(self, X: NumpyArray):
        """
        Method used to check the parameters passed

        Parameters
        ----------
        X: Numpy_Array
            Dataset to be validate parameters against
        """
        metric_str = None
        if isinstance(self.metric, str):
            metric_str = self.metric

        averaging_algorithm = self.averaging_algorithm
        if isinstance(averaging_algorithm, str):
            if metric_str is not None and averaging_algorithm == "auto":
                if metric_str == "dtw":
                    averaging_algorithm = "dba"
                else:
                    averaging_algorithm = "mean"

            self._averaging_algorithm = TimeSeriesKMeans._averaging_algorithm_dict[
                averaging_algorithm
            ]
        super(TimeSeriesKMeans, self)._check_params(X)
