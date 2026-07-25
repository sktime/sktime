"""Metrics to assess performance on the clustering task."""

import numpy as np

from sktime.base import BaseObject
from sktime.distances import pairwise_distance


class BaseClusterMetric(BaseObject):
    """Base class for cluster evaluation metrics."""

    def evaluate(self, X, labels):
        """
        Evaluate the quality of clustering performed on data X for given labels.

        Parameters
        ----------
        X : panel-like object
            The input time series data.
        labels : array-like of int
            Cluster labels for each time series in X.

        Returns
        -------
        score : float
            The computed metric score.
        """
        raise NotImplementedError()


class TimeSeriesSilhouetteScore(BaseClusterMetric):
    """
    Silhouette score for time series clustering.

    This implementation computes the silhouette score by first calculating the
    pairwise distance matrix using a specified time series distance metric.
    """

    def __init__(self, metric="euclidean", **metric_params):
        """
        Initialize the silhouette score metric.

        Parameters
        ----------
        metric : str, default="euclidean"
            The distance metric to use.
        **metric_params : dict
            Additional keyword arguments to pass to the distance function.
        """
        self.metric = metric
        self.metric_params = metric_params.copy()
        super().__init__()

    def evaluate(self, X, labels):
        """
        Compute the silhouette score for time series clustering.

        Parameters
        ----------
        X : panel-like object
            The input time series data.
            It must be in the format expected by pairwise_distance.
        labels : array-like of int
            Cluster labels for each time series in X.

        Returns
        -------
        score : float
            The mean silhouette score over all time series.
        """
        distance_matrix = pairwise_distance(X, metric=self.metric, **self.metric_params)
        n = len(labels)
        unique_labels = np.unique(labels)

        # If there's only one cluster, the silhouette score is not defined.
        if len(unique_labels) < 2:
            return 0.0

        silhouette_values = np.zeros(n)

        for i in range(n):
            same_cluster = np.where(labels == labels[i])[0]
            same_cluster = same_cluster[same_cluster != i]  # Exclude self
            a = (
                np.mean(distance_matrix[i, same_cluster])
                if same_cluster.size > 0
                else 0.0
            )

            b = np.inf
            for label in unique_labels:
                if label == labels[i]:
                    continue
                other_cluster = np.where(labels == label)[0]
                if other_cluster.size > 0:
                    b = min(b, np.mean(distance_matrix[i, other_cluster]))

            silhouette_values[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

        return np.mean(silhouette_values)
