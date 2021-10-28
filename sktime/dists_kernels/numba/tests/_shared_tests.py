# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from numba import njit

from sktime.dists_kernels.numba.distances.base import DistanceCallable, NumbaDistance
from sktime.dists_kernels.tests._utils import create_test_distance_numpy


def _test_metric_parameters(distance_func: Callable):
    @njit()
    def _standalone_numba_distance(x, y) -> float:
        return 5.0

    class _ValidTestClass(NumbaDistance):
        @staticmethod
        @njit()
        def _numba_distance(x, y) -> float:
            return _standalone_numba_distance(x, y)

        def _distance_factory(
            self, x: np.ndarray, y: np.ndarray, **kwargs: dict
        ) -> DistanceCallable:
            return _ValidTestClass._numba_distance

    x_numpy = create_test_distance_numpy(10, 10)
    y_numpy = create_test_distance_numpy(10, 10, random_state=2)

    assert distance_func(x_numpy, y_numpy, metric=_ValidTestClass()) == 5.0
    assert distance_func(x_numpy, y_numpy, metric=_standalone_numba_distance) == 5


def _test_incorrect_parameters(distance_func: Callable):
    """Test to ensure correct errors thrown."""
    numpy_x = create_test_distance_numpy(10, 10)
    numpy_y = create_test_distance_numpy(10, 10, random_state=2)

    df_x = pd.DataFrame(numpy_x)

    series_x = df_x.iloc[0]

    numpy_4d = np.array([[[[1, 2, 3]]]])

    class _InvalidTestClass:
        @staticmethod
        @njit()
        def _numba_distance(x, y, **kwargs) -> float:
            return 5.0

        def _distance_factory(
            self, x: np.ndarray, y: np.ndarray, **kwargs: dict
        ) -> DistanceCallable:
            return self._numba_distance

    with pytest.raises(ValueError):  # Invalid metric string
        distance_func(numpy_x, numpy_y, metric="fake")

    with pytest.raises(ValueError):  # Invalid x type as df
        distance_func(df_x, numpy_y, metric="euclidean")

    with pytest.raises(ValueError):  # Invalid x type as series
        distance_func(series_x, numpy_y, metric="euclidean")

    with pytest.raises(ValueError):  # Invalid y type as df
        distance_func(numpy_x, df_x, metric="euclidean")

    with pytest.raises(ValueError):  # Invalid y as series
        distance_func(numpy_x, series_x, metric="euclidean")

    with pytest.raises(ValueError):  # Invalid dimensions x
        distance_func(numpy_4d, numpy_y, metric="euclidean")

    with pytest.raises(ValueError):  # Invalid dimensions y
        distance_func(numpy_x, numpy_4d, metric="euclidean")

    with pytest.raises(ValueError):  # Object that doesn't inherit NumbaDistance
        distance_func(numpy_x, numpy_y, metric=_InvalidTestClass())
