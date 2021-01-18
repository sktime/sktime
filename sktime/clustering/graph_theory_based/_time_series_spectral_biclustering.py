# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["TimeSeriesSpectralBiclustering"]

from sktime.clustering._cluster import Cluster
from sktime.clustering.types import Metric_Parameter
from sklearn.cluster import SpectralBiclustering


class TimeSeriesSpectralBiclustering(Cluster):
    """
    Kmeans clustering algorithm that is built upon the scikit learns
    implementation
    """

    def __init__(
        self,
        n_clusters: int = 4,
        method: str = "bistochastic",
        n_components: int = 6,
        n_best: int = 3,
        svd_method: str = "randomized",
        n_svd_vecs: int = None,
        mini_batch: bool = False,
        init: any = "k-means++",
        n_init: int = 10,
        random_state: any = None,
        metric: Metric_Parameter = None,
    ):
        super().__init__(
            metric=metric,
        )
        self.model = SpectralBiclustering(
            n_clusters=n_clusters,
            method=method,
            n_components=n_components,
            n_best=n_best,
            svd_method=svd_method,
            n_svd_vecs=n_svd_vecs,
            mini_batch=mini_batch,
            init=init,
            n_init=n_init,
            random_state=random_state,
        )
