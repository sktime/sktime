# -*- coding: utf-8 -*-
from sktime.dists_kernels.numba_aligners.dtw.dtw_cost_matrix import (
    dtw_cost_matrix_alignment,
    numba_dtw_cost_matrix_distance_factory,
)
from sktime.dists_kernels.numba_aligners.dtw.dtw_path import dtw_path_alignment

NUMBA_ALIGNERS = [
    (
        "dtw_based cost matrix alignment",
        dtw_cost_matrix_alignment,
        numba_dtw_cost_matrix_distance_factory,
    ),
    ("dtw_based path", dtw_path_alignment, numba_dtw_cost_matrix_distance_factory),
]
