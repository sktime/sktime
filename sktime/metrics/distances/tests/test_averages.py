# -*- coding: utf-8 -*-
from typing import List

from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.data_processing import from_nested_to_3d_numpy
from sktime.dists_kernels.distances.averages import _medoids


def _create_test_ts_matrix(dimensions: List[int]):
    nested, _ = make_classification_problem(
        n_instances=dimensions[0],
        n_columns=dimensions[1],
        n_timepoints=dimensions[2],
        n_classes=1,
    )
    numpy_ts = from_nested_to_3d_numpy(nested)
    matrix = numpy_ts
    return matrix


def test_averages():
    matrix = _create_test_ts_matrix([100, 10, 10])
    test = _medoids()
    test.average(matrix)
