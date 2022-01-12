# -*- coding: utf-8 -*-
"""Test function of elastic distance nearest neighbour classifiers."""
from sktime.classification.distance_based._time_series_neighbors import (
    KNeighborsTimeSeriesClassifier,
)
from sktime.datasets import load_unit_test

distance_functions = [
    "euclidean",
    "dtw",
    "wdtw",
    # "msm",
    "erp",
    "lcss",
]

# expected correct on test set using default parameters.
expected_correct = {
    "euclidean": 19,
    "dtw": 21,
    "wdtw": 21,
    "msm": 20,
    "erp": 12,
    "lcss": 12,
}


def test_knn_on_unit_test():
    """Test function for elastic knn, to be reinstated soon."""
    # load arrowhead data for unit tests
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    for i in range(0, len(distance_functions)):
        knn = KNeighborsTimeSeriesClassifier(
            distance=distance_functions[i],
        )
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        correct = 0
        for j in range(0, len(pred)):
            if pred[j] == y_test[j]:
                correct = correct + 1
        assert correct == expected_correct[distance_functions[i]]


def test_knn_bounding_matrix():
    """Test knn with custom bounding parameters."""
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    for i in range(0, len(distance_functions)):
        knn = KNeighborsTimeSeriesClassifier(
            distance=distance_functions[i], distance_params={"window": 0.5}
        )
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        correct = 0
        for j in range(0, len(pred)):
            if pred[j] == y_test[j]:
                correct = correct + 1
        assert correct == expected_correct[distance_functions[i]]
