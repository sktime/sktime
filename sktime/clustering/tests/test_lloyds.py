# -*- coding: utf-8 -*-
"""Tests for time series Lloyds partitioning."""
import numpy as np

from sktime.clustering.partitioning._lloyds import TimeSeriesLloyds
from sktime.distances.tests._utils import create_test_distance_numpy

dataset_name = "Beef"


class _test_class(TimeSeriesLloyds):
    def _compute_new_cluster_centers(
        self, X: np.ndarray, assignment_indexes: np.ndarray
    ) -> np.ndarray:
        return self.cluster_centers_

    def __init__(self):
        super(_test_class, self).__init__(random_state=1)
        pass


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
        np.array([2, 1, 1, 3, 2, 2, 2, 2, 6, 2, 2, 6, 6, 6, 2, 4, 3, 0, 6, 4]),
    )
