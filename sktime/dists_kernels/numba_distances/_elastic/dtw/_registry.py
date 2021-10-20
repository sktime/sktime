# -*- coding: utf-8 -*-
from sktime.dists_kernels.numba_distances._elastic.dtw.dtw_distance import (
    dtw_distance,
    numba_dtw_distance_factory,
)
from sktime.dists_kernels.numba_distances._elastic.dtw.dtw_cost_matrix import (
    dtw_cost_matrix_alignment,
    numba_dtw_cost_matrix_distance_factory,
)

DTW_DISTANCES = [("dtw distance", dtw_distance, numba_dtw_distance_factory)]

DTW_ALIGNERS = [
    (
        "dtw cost matrix",
        dtw_cost_matrix_alignment,
        numba_dtw_cost_matrix_distance_factory,
    )
]
