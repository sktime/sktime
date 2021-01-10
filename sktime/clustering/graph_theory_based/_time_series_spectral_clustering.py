# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["TimeSeriesSpectralClustering"]

from typing import Union
from sktime.clustering._cluster import Cluster, distance_parameter
from sklearn.cluster import SpectralClustering


class TimeSeriesSpectralClustering(Cluster):
    """
    Kmeans clustering algorithm that is built upon the scikit learns
    implementation
    """

    def __init__(
        self,
        n_clusters: int = 8,
        eigen_solver: Union[str, None] = None,
        n_components: Union[int, None] = None,
        random_state: any = None,
        n_init: int = 10,
        gamma: float = 1.0,
        affinity: any = "rbf",
        n_neighbors: int = "rbf",
        eigen_tol: float = 0.0,
        assign_labels: str = "kmeans",
        degree: float = 3,
        coef0: float = 1,
        kernel_params: any = None,
        n_jobs: Union[int, None] = None,
        distance: distance_parameter = None,
    ):
        super().__init__(
            model=SpectralClustering(
                n_clusters=n_clusters,
                eigen_solver=eigen_solver,
                n_components=n_components,
                random_state=random_state,
                n_init=n_init,
                gamma=gamma,
                affinity=affinity,
                n_neighbors=n_neighbors,
                eigen_tol=eigen_tol,
                assign_labels=assign_labels,
                degree=degree,
                coef0=coef0,
                kernel_params=kernel_params,
                n_jobs=n_jobs,
            ),
            distance=distance,
        )
