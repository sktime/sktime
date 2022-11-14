# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

from sktime.distances.distance_rework_two._base import BaseLocalDistance

LocalDistanceParam = Union[np.ndarray, float]


class _SquaredDistance(BaseLocalDistance):

    _numba_distance = True
    _cache = True
    _fastmath = True

    @staticmethod
    def _distance(x: np.ndarray, y: np.ndarray, *args) -> float:
        dims = x.shape[0]
        timepoints = x.shape[1]
        distance = 0

        for i in range(dims):
            for j in range(timepoints):
                distance += (x[i, j] - y[i, j]) ** 2
        return distance

    @staticmethod
    def _local_distance(x: float, y: float, *args) -> float:
        return (x - y) ** 2

    @staticmethod
    def _result_process(result: float):
        return result * 2
