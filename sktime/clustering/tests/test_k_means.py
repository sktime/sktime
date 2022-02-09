# -*- coding: utf-8 -*-
"""Tests for time series k-means."""
import numpy as np
from sklearn import metrics

from sktime.clustering._k_means import TimeSeriesKMeans
from sktime.datasets import load_basic_motions

expected_results = {
    "mean": [
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        3,
        2,
        2,
        2,
        2,
        0,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]
}

expected_train_result = {"mean": 0.4846153846153846}

expected_score = {"mean": 0.3153846153846154}

expected_iters = {"mean": 4}

expected_labels = {
    "mean": [
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        0,
        0,
        1,
        1,
        3,
        1,
        0,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        0,
        2,
        2,
        2,
        0,
        2,
        2,
    ]
}


def test_kmeans():
    """Test implementation of Kmeans."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kmeans = TimeSeriesKMeans(
        averaging_method="mean",
        random_state=1,
        n_init=2,
        n_clusters=4,
        init_algorithm="kmeans++",
        metric="dtw",
    )
    train_predict = kmeans.fit_predict(X_train)
    train_mean_score = metrics.rand_score(y_train, train_predict)

    test_mean_result = kmeans.predict(X_test)
    mean_score = metrics.rand_score(y_test, test_mean_result)
    proba = kmeans.predict_proba(X_test)

    assert np.array_equal(test_mean_result, expected_results["mean"])
    assert mean_score == expected_score["mean"]
    assert train_mean_score == expected_train_result["mean"]
    assert kmeans.n_iter_ == expected_iters["mean"]
    assert np.array_equal(kmeans.labels_, expected_labels["mean"])
    assert isinstance(kmeans.cluster_centers_, np.ndarray)
    assert proba.shape == (40, 4)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1
