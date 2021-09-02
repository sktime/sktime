# -*- coding: utf-8 -*-
# import numpy as np
#
# from sktime.metrics.distances._dtw_based import dtw,
# LowerBounding, dtw_and_cost_matrix
# from sktime.metrics.distances.tests.utils import _create_test_ts_distances
#
#
# def test_dtw_distance():
#     x, y = _create_test_ts_distances([4, 4])
#     dtw(x, y, lower_bounding=1)
#     dtw(x, y, lower_bounding=2)
#     dtw(x, y, lower_bounding=3)
#
#
# def test_dtw_with_cost_matrix_distance():
#     x, y = _create_test_ts_distances([10, 10])
#     dtw_and_cost_matrix(x, y, lower_bounding=1)
#
#
# def test_lower_bounding():
#     x, y = _create_test_ts_distances([10, 10])
#     no_constraints = LowerBounding.NO_BOUNDING
#
#     assert np.array_equal(
#         no_constraints.create_bounding_matrix(x, y), np.zeros((x.shape[0],
#         y.shape[0]))
#     )
#
#     sakoe_chiba = LowerBounding.SAKOE_CHIBA
#
#     sakoe_chiba.create_bounding_matrix(x, y, sakoe_chiba_window_radius=10)
#
#     itakura_parallelogram = LowerBounding.ITAKURA_PARALLELOGRAM
#     itakura_parallelogram.create_bounding_matrix(x, y)
