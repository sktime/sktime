# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["TimeSeriesMeanShift"]


from sktime.clustering._cluster import Cluster, distance_parameter
from sklearn.cluster import MeanShift


class TimeSeriesMeanShift(Cluster):
    """
    Kmeans clustering algorithm that is built upon the scikit learns
    implementation
    """

    def __init__(
        self,
        bandwidth: float = None,
        seeds: any = None,
        bin_seeding: bool = False,
        min_bin_freq: int = 1,
        cluster_all: bool = True,
        n_jobs: int = None,
        max_iter: int = 300,
        distance: distance_parameter = None,
    ):
        super().__init__(
            model=MeanShift(
                bandwidth=bandwidth,
                seeds=seeds,
                bin_seeding=bin_seeding,
                min_bin_freq=min_bin_freq,
                cluster_all=cluster_all,
                n_jobs=n_jobs,
                max_iter=max_iter,
            ),
            distance=distance,
        )
