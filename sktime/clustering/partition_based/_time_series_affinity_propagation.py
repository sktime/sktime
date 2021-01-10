# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["Cluster"]


from sktime.clustering._cluster import Cluster, distance_parameter
from sklearn.cluster import AffinityPropagation


class TimeSeriesAffinityPropagation(Cluster):
    """
    Kmeans clustering algorithm that is built upon the scikit learns
    implementation
    """

    def __init__(
        self,
        damping: float = 0.5,
        max_iter: int = 200,
        copy: bool = True,
        preference=None,
        affinity: str = "euclidean",
        verbose: bool = False,
        random_state=0,
        distance: distance_parameter = None,
    ):
        Cluster.__init__(
            self,
            model=AffinityPropagation(
                damping=damping,
                max_iter=max_iter,
                copy=copy,
                preference=preference,
                affinity=affinity,
                verbose=verbose,
                random_state=random_state,
            ),
            distance=distance,
        )
