# -*- coding: utf-8 -*-
from sktime.dists_kernels.numba_distances._elastic.squared_distance import (
    squared_distance,
    pairwise_squared_distance,
)
from sktime.dists_kernels.numba_distances._elastic.euclidean_distance import (
    euclidean_distance,
    pairwise_euclidean_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw_based.dtw_distance import (
    dtw_distance,
    pairwise_dtw_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw_based.ddtw_distance import (
    ddtw_distance,
    pairwise_ddtw_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw_based.wdtw_distance import (
    wdtw_distance,
    pairwise_wdtw_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw_based.wddtw_distance import (
    wddtw_distance,
    pairwise_wddtw_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw_based.lcss_distance import (
    lcss_distance,
    pairwise_lcss_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw_based.edr_distance import (
    edr_distance,
    pairwise_edr_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw_based.erp_distance import (
    erp_distance,
    pairwise_erp_distance,
)


NUMBA_DISTANCES = [
    ("squared distance", squared_distance, pairwise_squared_distance),
    ("euclidean distance", euclidean_distance, pairwise_euclidean_distance),
    ("dtw_based distance", dtw_distance, pairwise_dtw_distance),
    ("ddtw distance", ddtw_distance, pairwise_ddtw_distance),
    ("wdtw distance", wdtw_distance, pairwise_wdtw_distance),
    ("wddtw distance", wddtw_distance, pairwise_wddtw_distance),
    ("lcss distance", lcss_distance, pairwise_lcss_distance),
    ("edr distance", edr_distance, pairwise_edr_distance),
    ("erp distance", erp_distance, pairwise_erp_distance),
]
