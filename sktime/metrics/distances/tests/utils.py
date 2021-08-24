# -*- coding: utf-8 -*-
from typing import List

from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.data_processing import from_nested_to_3d_numpy


def _create_test_ts_distances(dimensions: List[int]):
    nested, _ = make_classification_problem(
        n_instances=2,
        n_columns=dimensions[0],
        n_timepoints=10,
        n_classes=1,
        random_state=1,
    )
    numpy_ts = from_nested_to_3d_numpy(nested)
    x = numpy_ts[0]
    y = numpy_ts[1]
    return x, y


def _create_test_ts_matrix(n_instances, n_columns, n_timepoints):
    nested, _ = make_classification_problem(
        n_instances=n_instances,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        n_classes=1,
    )
    numpy_ts = from_nested_to_3d_numpy(nested)
    return numpy_ts
