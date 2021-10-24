# -*- coding: utf-8 -*-
from sktime.dists_kernels.numba_distances._elastic.squared_distance import (
    squared_distance,
    pairwise_squared_distance,
)
from sktime.dists_kernels.numba_distances._elastic.euclidean_distance import (
    euclidean_distance,
    pairwise_euclidean_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw.dtw_distance import (
    dtw_distance,
    pairwise_dtw_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw.ddtw_distance import (
    ddtw_distance,
    pairwise_ddtw_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw.wdtw_distance import (
    wdtw_distance,
    pairwise_wdtw_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw.wddtw_distance import (
    wddtw_distance,
    pairwise_wddtw_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw.lcss_distance import (
    lcss_distance,
    pairwise_lcss_distance,
)


NUMBA_DISTANCES = [
    ("squared distance", squared_distance, pairwise_squared_distance),
    ("euclidean distance", euclidean_distance, pairwise_euclidean_distance),
    ("dtw distance", dtw_distance, pairwise_dtw_distance),
    ("ddtw distance", ddtw_distance, pairwise_ddtw_distance),
    ("wdtw distance", wdtw_distance, pairwise_wdtw_distance),
    ("wddtw distance", wddtw_distance, pairwise_wddtw_distance),
    ("lcss distance", lcss_distance, pairwise_lcss_distance),
]
