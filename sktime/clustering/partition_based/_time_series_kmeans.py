# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"

from sktime.clustering._cluster import Cluster, Metric_Parameter
from sklearn.cluster import KMeans


class TimeSeriesKMeans(Cluster):
    """
    Kmeans clustering algorithm that is built upon the scikit learns
    implementation
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=1,
        random_state=None,
        copy_x=True,
        metric: Metric_Parameter = None
    ):
        """
        Constructor for TimeSeriesKMeans
        """
        super().__init__(
            metric=metric,
        )
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm="auto",
        )
