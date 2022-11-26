# -*- coding: utf-8 -*-
"""Test function of elastic distance nearest neighbour classifiers."""
import numpy as np
import pytest

from sktime.alignment.dtw_python import AlignerDTW
from sktime.classification.distance_based._time_series_neighbors import (
    KNeighborsTimeSeriesClassifier,
)
from sktime.datasets import load_unit_test
from sktime.utils.validation._dependencies import _check_estimator_deps

distance_functions = [
    "euclidean",
    "dtw",
    "wdtw",
    "msm",
    "erp",
    "lcss",
    "edr",
]

# expected correct on test set using default parameters.
expected_correct = {
    "euclidean": 19,
    "dtw": 21,
    "wdtw": 21,
    "msm": 20,
    "erp": 19,
    "lcss": 12,
    "edr": 20,
}

# expected correct on test set using window params.
expected_correct_window = {
    "euclidean": 19,
    "dtw": 21,
    "wdtw": 21,
    "msm": 10,
    "erp": 19,
    "edr": 20,
    "lcss": 12,
}


@pytest.mark.parametrize("distance_key", distance_functions)
def test_knn_on_unit_test(distance_key):
    """Test function for elastic knn, to be reinstated soon."""
    # load arrowhead data for unit tests
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    knn = KNeighborsTimeSeriesClassifier(
        distance=distance_key,
    )
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    correct = 0
    for j in range(0, len(pred)):
        if pred[j] == y_test[j]:
            correct = correct + 1
    assert correct == expected_correct[distance_key]


@pytest.mark.parametrize("distance_key", distance_functions)
def test_knn_bounding_matrix(distance_key):
    """Test knn with custom bounding parameters."""
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    knn = KNeighborsTimeSeriesClassifier(
        distance=distance_key, distance_params={"window": 0.5}
    )
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    correct = 0
    for j in range(0, len(pred)):
        if pred[j] == y_test[j]:
            correct = correct + 1
    assert correct == expected_correct_window[distance_key]


@pytest.mark.skipif(
    not _check_estimator_deps(AlignerDTW, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_knn_with_aligner():
    """Tests KNN classifer with alignment distance on unequal length data."""
    from sktime.dists_kernels.compose_from_align import DistFromAligner
    from sktime.utils._testing.hierarchical import _make_hierarchical

    X = _make_hierarchical((3,), min_timepoints=5, max_timepoints=10, random_state=0)
    y = np.array([0, 1, 1])

    dtw_dist = DistFromAligner(AlignerDTW())
    clf = KNeighborsTimeSeriesClassifier(distance=dtw_dist)

    clf.fit(X, y)


def test_knn_with_aggrdistance():
    """Tests KNN classifer with alignment distance on unequal length data."""
    from sktime.dists_kernels import AggrDist, ScipyDist
    from sktime.utils._testing.hierarchical import _make_hierarchical

    X = _make_hierarchical((3,), min_timepoints=5, max_timepoints=10, random_state=0)
    y = np.array([0, 1, 1])

    eucl_dist = ScipyDist()
    aggr_dist = AggrDist(eucl_dist)
    clf = KNeighborsTimeSeriesClassifier(distance=aggr_dist)

    clf.fit(X, y)
