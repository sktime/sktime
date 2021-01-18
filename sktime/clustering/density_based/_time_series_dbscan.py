# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["TimeSeriesDBSCAN"]

from typing import List
from sktime.clustering._cluster import Cluster
from sklearn.cluster import DBSCAN
from sktime.clustering.types import Metric_Parameter


class TimeSeriesDBSCAN(Cluster):
    """
    Time series affinity propagation clustering class
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric_params: dict = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: float = None,
        n_jobs: int = None,
        metric: Metric_Parameter = "euclidean",
    ):
        super().__init__(
            metric=metric,
        )
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        )

    def model_built_in_metrics(self) -> List[str]:
        return ["euclidean"]
