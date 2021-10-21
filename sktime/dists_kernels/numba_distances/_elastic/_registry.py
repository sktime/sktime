# -*- coding: utf-8 -*-
from sktime.dists_kernels.numba_distances._elastic.squared_distance import (
    squared_distance,
    numba_squared_distance_factory,
)
from sktime.dists_kernels.numba_distances._elastic.euclidean_distance import (
    euclidean_distance,
    pairwise_euclidean_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw.dtw_distance import (
    dtw_distance,
    numba_dtw_distance_factory,
)
from sktime.dists_kernels.numba_distances._elastic.dtw.ddtw_distance import (
    ddtw_distance,
)


NUMBA_DISTANCES = [
    # ("squared distance", squared_distance, numba_squared_distance_factory),
    ("euclidean distance", euclidean_distance, pairwise_euclidean_distance),
    # ("dtw distance", dtw_distance, numba_dtw_distance_factory),
    # ("ddtw distance", ddtw_distance, numba_dtw_distance_factory)
]
