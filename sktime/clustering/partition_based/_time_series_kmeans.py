# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"

from sktime.clustering._cluster import Cluster
from sktime.clustering.types import Metric_Parameter
from sklearn.cluster import KMeans


def check_kmeans_parameters(
    n_clusters, init, n_init, max_iter, tol, verbose, random_state, copy_x
) -> tuple:
    """
    Method that is used to check the parameters passed to construct a sklearn
    kmeans classier
    """
    if verbose is False:
        verbose = 0
    elif verbose is True:
        verbose = 1
    return (n_clusters, init, n_init, max_iter, tol, verbose, random_state, copy_x)


class TimeSeriesKMeans(Cluster):
    """
    Kmeans clustering algorithm that is built upon the scikit learns
    implementation
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: any = None,
        copy_x: bool = True,
        metric: Metric_Parameter = None
    ):
        """
        Constructor for TimeSeriesKMeans
        """
        super().__init__(
            metric=metric,
        )
        params = check_kmeans_parameters(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
        )
        self.model = KMeans(*params)
