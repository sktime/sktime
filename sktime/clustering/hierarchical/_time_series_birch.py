# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["TimeSeriesBirch"]

from sktime.clustering._cluster import Cluster
from sklearn.cluster import Birch
from sktime.clustering.types import Metric_Parameter


class TimeSeriesBirch(Cluster):
    """
    Note: compute_distances throwing error when passed as parameter
    """

    def __init__(
        self,
        threshold: float = 0.5,
        branching_factor: int = 50,
        n_clusters: any = 3,
        compute_labels: bool = True,
        copy: bool = True,
        metric: Metric_Parameter = None,
    ):
        super().__init__(metric=metric)
        self.model = Birch(
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters,
            compute_labels=compute_labels,
            copy=copy,
        )
