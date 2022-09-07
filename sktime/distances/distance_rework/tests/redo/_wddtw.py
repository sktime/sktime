# -*- coding: utf-8 -*-
from typing import Callable
import numpy as np

from sktime.distances.distance_rework.tests.redo import _WdtwDistance
from sktime.distances.distance_rework.tests.redo._ddtw import average_of_slope


class _WddtwDistance(_WdtwDistance):
    _has_cost_matrix = True
    _numba_distance = True
    _cache = True
    _fastmath = True

    def _preprocessing_time_series_callback(
            self, **kwargs
    ) -> Callable[[np.ndarray], np.ndarray]:
        if 'compute_derivative' in kwargs:
            return kwargs['compute_derivative']
        else:
            return average_of_slope
