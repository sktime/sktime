# -*- coding: utf-8 -*-
"""Tests for time series k-means."""
import numpy as np
from sklearn import metrics

from sktime.clustering._k_means import TimeSeriesKMeans
from sktime.datasets import load_UCR_UEA_dataset

dataset_name = "Beef"

expected_results = {
    "mean": [
        1,
        7,
        7,
        7,
        7,
        6,
        4,
        2,
        1,
        5,
        3,
        2,
        4,
        2,
        1,
        1,
        2,
        2,
        1,
        3,
        5,
        1,
        3,
        3,
        4,
        0,
        5,
        5,
        2,
        0,
    ]
}

expected_score = {"mean": 0.7862068965517242}

expected_iters = {"mean": 300}

expected_labels = {
    "mean": [
        6,
        7,
        7,
        7,
        7,
        6,
        4,
        3,
        5,
        5,
        2,
        2,
        4,
        2,
        4,
        4,
        0,
        2,
        1,
        3,
        5,
        1,
        3,
        3,
        4,
        0,
        5,
        5,
        3,
        0,
    ]
}


def test_kmeans():
    """Test implementation of Kmeans."""
    X_train, y_train = load_UCR_UEA_dataset(
        dataset_name, split="train", return_X_y=True
    )
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

    kmeans = TimeSeriesKMeans(averaging_method="mean", random_state=1)
    kmeans.fit(X_train)
    test_mean_result = kmeans.predict(X_test)
    mean_score = metrics.rand_score(y_test, test_mean_result)

    assert np.array_equal(test_mean_result, expected_results["mean"])
    assert mean_score == expected_score["mean"]
    assert kmeans.n_iter_ == 300
    assert np.array_equal(kmeans.labels_, expected_labels["mean"])
    assert isinstance(kmeans.cluster_centers_, np.ndarray)
