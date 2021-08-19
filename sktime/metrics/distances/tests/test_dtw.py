# -*- coding: utf-8 -*-
import numpy as np
from typing import List

from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.data_processing import from_nested_to_3d_numpy
from sktime.metrics.distances._dtw_based import dtw, LowerBounding, dtw_and_cost_matrix


def _create_test_ts(dimensions: List[int]):
    nested, _ = make_classification_problem(
        n_instances=2, n_columns=dimensions[0], n_timepoints=10, n_classes=1
    )
    numpy_ts = from_nested_to_3d_numpy(nested)
    x = numpy_ts[0]
    nested, _ = make_classification_problem(
        n_instances=2, n_columns=dimensions[1], n_timepoints=10, n_classes=1
    )
    numpy_ts = from_nested_to_3d_numpy(nested)
    y = numpy_ts[0]
    return x, y


def test_dtw_distance():
    x, y = _create_test_ts([4, 4])
    dtw(x, y, lower_bounding=1)
    dtw(x, y, lower_bounding=2)
    dtw(x, y, lower_bounding=3)


def test_dtw_with_cost_matrix_distance():
    x, y = _create_test_ts([10, 10])
    dtw_and_cost_matrix(x, y, lower_bounding=1)


def test_lower_bounding():
    x, y = _create_test_ts([10, 10])
    no_constraints = LowerBounding.NO_BOUNDING

    assert np.array_equal(
        no_constraints.create_bounding_matrix(x, y), np.zeros((x.shape[0], y.shape[0]))
    )

    sakoe_chiba = LowerBounding.SAKOE_CHIBA

    sakoe_chiba.create_bounding_matrix(x, y, sakoe_chiba_window_radius=10)

    itakura_parallelogram = LowerBounding.ITAKURA_PARALLELOGRAM
    itakura_parallelogram.create_bounding_matrix(x, y)
