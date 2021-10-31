# -*- coding: utf-8 -*-
# The key string (i.e. 'euclidean') must be the same as the name in _registry
_expected_distance_results = {
    # Result structure:
    # [single value series, univariate series, multivariate series, multivariate panel]
    "squared": [25.0, 6.93261, 50.31911, 499.67497],
    "euclidean": [5.0, 2.63298, 7.09359, 70.56413],
    "dtw": [25.0, 2.18036, 46.68062, 463.98655],
    "ddtw": [0.0, 2.08848, 31.57625, 310.22540],
    "wdtw": [25.0, 2.18036, 46.68062, 463.98655],
}
