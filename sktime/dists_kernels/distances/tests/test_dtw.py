# -*- coding: utf-8 -*-
import numpy as np

from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.data_processing import from_nested_to_3d_numpy
from sktime.dists_kernels.distances.dtw import LowerBounding


def test_dtw_distance():
    pass


def test_lower_bounding():
    dimensions = [8, 50]
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

    no_constraints = LowerBounding.NO_BOUNDING

    assert np.array_equal(
        no_constraints.create_bounding_matrix(x, y), np.zeros((x.shape[0], y.shape[0]))
    )

    sakoe_chiba = LowerBounding.SAKOE_CHIBA

    sakoe_chiba.create_bounding_matrix(x, y)

    itakura_parallelogram = LowerBounding.ITAKURA_PARALLELOGRAM
    itakura_parallelogram.create_bounding_matrix(x, y)
