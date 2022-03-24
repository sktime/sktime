# -*- coding: utf-8 -*-
"""Perform benchmarking timing experiments to help assess impact of code changes.

These are not formal tests. Instead, they form a means of manual sanity checks to
ensure that changes to algorithms do not result in significant slow down,
or conversely to assess whether changes give significant speed up.
"""
__author__ = ["TonyBagnall"]

import numpy as np

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids


def distance_timing():
    """Time distance functions for increased series length."""
    k_medoids = TimeSeriesKMedoids(
        n_clusters=5,  # Number of desired centers
        init_algorithm="forgy",  # Center initialisation technique
        max_iter=10,  # Maximum number of iterations for refinement on training set
        metric="dtw",  # Distance metric to use
        random_state=1,
    )
    k_means = TimeSeriesKMeans(
        n_clusters=5,  # Number of desired centers
        init_algorithm="forgy",  # Center initialisation technique
        max_iter=10,  # Maximum number of iterations for refinement on training set
        metric="dtw",  # Distance metric to use
        random_state=1,
    )
    for i in range(100):
        first = np.ndarray((100, 100 * (i + 1)))
        k_means.fit()
        k_medoids.fit()
