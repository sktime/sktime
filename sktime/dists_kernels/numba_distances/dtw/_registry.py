# -*- coding: utf-8 -*-
from sktime.dists_kernels.numba_distances.dtw.dtw import (
    dtw_distance,
    numba_dtw_distance_factory,
)

DTW_DISTANCES = [("dtw distance", dtw_distance, numba_dtw_distance_factory)]
