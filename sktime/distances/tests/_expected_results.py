# -*- coding: utf-8 -*-
# The key string (i.e. 'euclidean') must be the same as the name in _registry

_expected_distance_results = {
    # Result structure:
    # [single value series, univariate series, multivariate series, multivariate panel]
    "squared": [25.0, 6.93261, 50.31911],
    "euclidean": [5.0, 2.63298, 7.09359],
    "dtw": [25.0, 2.18037, 46.68062],
    "ddtw": [0.0, 2.08848, 31.57625],
    "wdtw": [12.5, 1.09018, 23.34031],
    "wddtw": [0.0, 1.04424, 15.78813],
    "lcss": [1.0, 0.1, 1.0],
    "edr": [1.0, 0.6, 1.0],
    "erp": [5.0, 5.03767, 20.78718],
}
