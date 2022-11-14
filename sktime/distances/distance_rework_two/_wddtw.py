# -*- coding: utf-8 -*-
from sktime.distances.distance_rework_two._ddtw import average_of_slope
from sktime.distances.distance_rework_two._wdtw import _WdtwDistance


class _WddtwDistance(_WdtwDistance):
    """Wddtw distance."""

    _numba_distance = True
    _cache = True
    _fastmath = True

    @staticmethod
    def _preprocess_timeseries(x, *args):
        return average_of_slope(x)
