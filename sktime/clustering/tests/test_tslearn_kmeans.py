# -*- coding: utf-8 -*-
"""Tests for time series k-means."""
from sklearn import metrics

from sktime.clustering.tslearn_kmeans import TslearnKmeans
from sktime.datasets import load_basic_motions


def test_kmeans():
    """Test implementation of Kmeans."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kmeans = TslearnKmeans(
        random_state=1,
        n_init=2,
        n_clusters=4,
        metric="dtw",
    )
    train_predict = kmeans.fit_predict(X_train)
    train_mean_score = metrics.rand_score(y_train, train_predict)

    test_mean_result = kmeans.predict(X_test)
    mean_score = metrics.rand_score(y_test, test_mean_result)
    proba = kmeans.predict_proba(X_test)
