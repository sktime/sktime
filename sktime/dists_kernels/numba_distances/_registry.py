# -*- coding: utf-8 -*-
from sktime.dists_kernels.numba_distances.squared_distance import (
    squared_distance,
    numba_squared_distance_factory,
)
from sktime.dists_kernels.numba_distances.euclidean_distance import (
    euclidean_distance,
    numba_euclidean_distance_factory,
)
from sktime.dists_kernels.numba_distances.dtw._registry import DTW_DISTANCES

NUMBA_DISTANCES = [
    ("squared distance", squared_distance, numba_squared_distance_factory),
    ("euclidean distance", euclidean_distance, numba_euclidean_distance_factory),
]

NUMBA_DISTANCES = NUMBA_DISTANCES + DTW_DISTANCES
