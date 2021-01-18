# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["TimeSeriesFeatureAgglomerative"]

from typing import Union, Callable
from sktime.clustering._cluster import Cluster
from sklearn.cluster import FeatureAgglomeration
from sktime.clustering.types import Metric_Parameter
import numpy as np


class TimeSeriesFeatureAgglomerative(Cluster):
    """
    Note: compute_distances throwing error when passed as parameter
    """

    def __init__(
        self,
        n_clusters: int = 2,
        affinity: str = "euclidean",
        memory: any = None,
        connectivity: any = None,
        compute_full_tree: Union[str, bool] = "auto",
        linkage: str = "ward",
        pooling_func: Callable = np.mean,
        distance_threshold: float = None,
        compute_distances: bool = False,
        metric: Metric_Parameter = None,
    ):
        super().__init__(metric=metric)
        self.model = FeatureAgglomeration(
            n_clusters=n_clusters,
            affinity=affinity,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            pooling_func=pooling_func,
            distance_threshold=distance_threshold,
        )
