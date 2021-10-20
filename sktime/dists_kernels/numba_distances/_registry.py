# -*- coding: utf-8 -*-
from sktime.dists_kernels.numba_distances._elastic.squared_distance import (
    squared_distance,
    numba_squared_distance_factory,
)
from sktime.dists_kernels.numba_distances._elastic.euclidean_distance import (
    euclidean_distance,
    numba_euclidean_distance_factory,
)
from sktime.dists_kernels.numba_distances._elastic.dtw._registry import DTW_DISTANCES

NUMBA_DISTANCES = [
    ("squared distance", squared_distance, numba_squared_distance_factory),
    ("euclidean distance", euclidean_distance, numba_euclidean_distance_factory),
]

NUMBA_DISTANCES = NUMBA_DISTANCES + DTW_DISTANCES
