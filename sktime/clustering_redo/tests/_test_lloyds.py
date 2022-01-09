# -*- coding: utf-8 -*-
import numpy as np

from sktime.clustering_redo.partitioning._lloyds import _Lloyds
from sktime.distances.tests._utils import create_test_distance_numpy

dataset_name = "Beef"


class _test_class(_Lloyds):
    def _compute_new_cluster_centers(
        self, X: np.ndarray, assignment_indexes: np.ndarray
    ) -> np.ndarray:
        return self.cluster_centers_

    def __init__(self):
        super(_test_class, self).__init__(random_state=1)
        pass


def test_lloyds():
    """Test implemention of Lloyds."""
    X_train = create_test_distance_numpy(20, 10, 10)

    lloyd = _test_class()
    lloyd.fit(X_train)
