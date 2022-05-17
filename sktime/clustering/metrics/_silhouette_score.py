# -*- coding: utf-8 -*-
from typing import List, Union

import numpy as np
from numpy.random import RandomState
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

from sktime.distances import pairwise_distance


def silhouette_score(
    X: np.ndarray,
    labels: List[int],
    metric: str = "dtw",
    metric_params: dict = None,
    sample_size: int = None,
    random_state: Union[int, RandomState] = None,
    **kwargs
):
    """Compute the mean Silhouette Coefficient of time series (cf.  [1]_ and [2]_).

    Read more in the `scikit-learn documentation
    <http://scikit-learn.org/stable/modules/clustering.html\
    #silhouette-coefficient>`_.

    Parameters
    ----------
    X : array [n_ts, n_ts] if metric == "precomputed", or, \
             [n_ts, sz, d] otherwise
        Array of pairwise distances between time series, or a time series
        dataset.
    labels : array, shape = [n_ts]
         Predicted labels for each time series.
    metric : string, callable or None (default: None)
        The metric to use when calculating distance between time series.
        Should be one of {'dtw', 'softdtw', 'euclidean'} or a callable distance
        function or None.
        If 'softdtw' is passed, a normalized version of Soft-DTW is used that
        is defined as `sdtw_(x,y) := sdtw(x,y) - 1/2(sdtw(x,x)+sdtw(y,y))`.
        If X is the distance array itself, use ``metric="precomputed"``.
        If None, dtw is used.
    sample_size : int or None (default: None)
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.
    metric_params : dict or None (default: None)
        Parameter values for the chosen metric.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` key passed in `metric_params` is overridden by
        the `n_jobs` argument.
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to randomly select a subset of samples.  If int,
        random_state is the seed used by the random number generator; If
        RandomState instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance used by
        `np.random`. Used when ``sample_size is not None``.
    kwargs: dict, default = None
        Additional kwargs.

    Returns
    -------
    float
        Mean Silhouette Coefficient for all samples.

    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    Examples
    --------
    >>> from sktime.distances.tests._utils import create_test_distance_numpy
    >>> np.random.seed(0)
    >>> X = create_test_distance_numpy(20, 10, 10)
    >>> labels = np.random.randint(2, size=20)
    >>> silhouette_score(X, labels, metric="dtw")
    -0.002578790325016581
    >>> silhouette_score(X, labels, metric="euclidean")
    -7.52631215894764e-05
    """
    if metric_params is None:
        metric_params = {}

    if metric == "precomputed":
        precomputed_distances = X
    else:
        precomputed_distances = pairwise_distance(X, metric=metric, **metric_params)

    return sklearn_silhouette_score(
        X=precomputed_distances,
        labels=labels,
        metric="precomputed",
        sample_size=sample_size,
        random_state=random_state,
        **kwargs
    )
