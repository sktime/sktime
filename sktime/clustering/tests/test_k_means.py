"""Tests for time series k-means."""

import numpy as np
import pytest
from sklearn import metrics

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.datasets import load_basic_motions
from sktime.tests.test_switch import run_test_for_class

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
    ],
    "dba": [1, 0, 1, 2, 2],
}

expected_train_result = {"mean": 0.4846153846153846, "dba": 0.1}

expected_score = {"mean": 0.3153846153846154, "dba": 0.2}

expected_iters = {"mean": 4, "dba": 2}

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
    ],
    "dba": [2, 1, 3, 0, 0],
}


@pytest.mark.skipif(
    not run_test_for_class(TimeSeriesKMeans),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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
        metric="euclidean",
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


@pytest.mark.skipif(
    not run_test_for_class(TimeSeriesKMeans),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_kmeans_dba():
    """Test implementation of Kmeans using dba."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    num_test_values = 5

    kmeans = TimeSeriesKMeans(
        averaging_method="dba",
        random_state=1,
        n_init=2,
        n_clusters=4,
        init_algorithm="kmeans++",
        metric="dtw",
    )
    train_predict = kmeans.fit_predict(X_train.head(num_test_values))
    train_mean_score = metrics.rand_score(y_train[0:num_test_values], train_predict)

    test_mean_result = kmeans.predict(X_test.head(num_test_values))
    mean_score = metrics.rand_score(y_test[0:num_test_values], test_mean_result)
    proba = kmeans.predict_proba(X_test.head(num_test_values))

    assert np.array_equal(test_mean_result, expected_results["dba"])
    assert mean_score == expected_score["dba"]
    assert train_mean_score == expected_train_result["dba"]
    assert kmeans.n_iter_ == expected_iters["dba"]
    assert np.array_equal(kmeans.labels_, expected_labels["dba"])
    assert isinstance(kmeans.cluster_centers_, np.ndarray)
    assert proba.shape == (5, 4)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1
