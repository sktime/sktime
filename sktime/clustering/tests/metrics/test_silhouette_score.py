# -*- coding: utf-8 -*-
"""Test for silhouette score."""
from numpy.random import RandomState

from sktime.clustering.metrics import silhouette_score
from sktime.distances.tests._utils import create_test_distance_numpy


def test_silhouette_score():
    """Test the silhouette score."""
    X = create_test_distance_numpy(10, 10, 10, random_state=1)

    labels = RandomState(1).randint(2, size=10)

    euclid_result = silhouette_score(X, labels, metric="euclidean", random_state=1)

    dtw_result = silhouette_score(X, labels, metric="dtw", random_state=1)

    dtw_params_result = silhouette_score(
        X, labels, metric="dtw", metric_params={"window": 0.2}, random_state=1
    )

    lcss_result = silhouette_score(X, labels, metric="lcss", random_state=1)

    assert round(euclid_result, 5) == -0.00798
    assert round(dtw_result, 5) == -0.03814
    assert round(dtw_params_result, 5) == -0.03878
    assert round(lcss_result, 5) == -0.02078
