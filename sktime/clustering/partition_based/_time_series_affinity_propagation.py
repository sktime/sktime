# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["TimeSeriesAffinityPropagation"]


from sktime.clustering._cluster import Cluster
from sklearn.cluster import AffinityPropagation
from sktime.clustering.types import Metric_Parameter


class TimeSeriesAffinityPropagation(Cluster):
    """
    Time series affinity propagation clustering class
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
        metric: Metric_Parameter = None,
    ):
        super().__init__(
            metric=metric,
        )
        self.model = AffinityPropagation(
            damping=damping,
            max_iter=max_iter,
            copy=copy,
            preference=preference,
            affinity=affinity,
            verbose=verbose,
            random_state=random_state,
        )
