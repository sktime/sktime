# -*- coding: utf-8 -*-
"""Tests for time series Lloyds partitioning."""
from typing import Callable

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from sktime.clustering.partitioning._lloyds import (
    TimeSeriesLloyds,
    _forgy_center_initializer,
    _kmeans_plus_plus,
    _random_center_initializer,
)
from sktime.datasets import load_arrow_head
from sktime.datatypes import convert_to
from sktime.distances.tests._utils import create_test_distance_numpy


class _test_class(TimeSeriesLloyds):
    def _compute_new_cluster_centers(
        self, X: np.ndarray, assignment_indexes: np.ndarray
    ) -> np.ndarray:
        return self.cluster_centers_

    def __init__(self):
        super(_test_class, self).__init__(random_state=1, n_init=2)


def test_lloyds():
    """Test implementation of Lloyds."""
    X_train = create_test_distance_numpy(20, 10, 10)
    X_test = create_test_distance_numpy(20, 10, 10, random_state=2)

    lloyd = _test_class()
    lloyd.fit(X_train)
    test_result = lloyd.predict(X_test)

    assert test_result.dtype is np.dtype("int64")

    assert np.array_equal(
        test_result,
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        ),
    )


CENTER_INIT_ALGO = [
    _kmeans_plus_plus,
    _random_center_initializer,
    _forgy_center_initializer,
]


@pytest.mark.parametrize("center_init_callable", CENTER_INIT_ALGO)
def test_center_init(center_init_callable: Callable[[np.ndarray], np.ndarray]):
    """Test center initialisation algorithms."""
    k = 5
    X, y = load_arrow_head(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = convert_to(X_train, "numpy3D")
    random_state = check_random_state(1)
    test_centers = center_init_callable(X_train, k, random_state)
    assert len(test_centers) == k
    assert len(np.unique(test_centers, axis=1)) == k
