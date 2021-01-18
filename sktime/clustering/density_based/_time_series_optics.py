# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["TimeSeriesOPTICS"]

from typing import List, Union
from sktime.clustering._cluster import Cluster
from sklearn.cluster import OPTICS
from sktime.clustering.types import Metric_Parameter
import numpy as np


class TimeSeriesOPTICS(Cluster):
    """
    Time series affinity propagation clustering class
    """

    def __init__(
        self,
        min_samples: Union[int, float] = 5,
        max_eps: float = np.inf,
        metric: Metric_Parameter = "minkowski",
        p: int = 2,
        metric_params: dict = None,
        cluster_method: str = "xi",
        eps: float = None,
        xi: float = 0.05,
        predecessor_correction: bool = True,
        min_cluster_size: Union[int, float] = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        n_jobs: int = None,
    ):
        super().__init__(
            metric=metric,
        )
        self.model = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            metric=metric,
            p=p,
            metric_params=metric_params,
            cluster_method=cluster_method,
            eps=eps,
            xi=xi,
            predecessor_correction=predecessor_correction,
            min_cluster_size=min_cluster_size,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
        )

    def model_built_in_metrics(self) -> List[str]:
        return ["minkowski"]
