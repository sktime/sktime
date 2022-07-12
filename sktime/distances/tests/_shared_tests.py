# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np
import pytest
from numba import njit

from sktime.distances.base import DistanceCallable, NumbaDistance
from sktime.distances.tests._utils import create_test_distance_numpy


def _test_metric_parameters(distance_func: Callable):
    """Test to ensure custom distances can be used."""

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

    class_result = distance_func(x_numpy, y_numpy, metric=_ValidTestClass())
    standalone_dist_result = distance_func(
        x_numpy, y_numpy, metric=_standalone_numba_distance
    )

    if isinstance(class_result, float) or class_result.shape == (1, 1):
        expected = 5.0
        assert class_result == expected, (
            f"Using a custom NumbaDistance did not produce the expected result. Ensure"
            f"custom NumbaDistances can be passed. Expected result {expected}, got "
            f"{class_result}"
        )

        assert standalone_dist_result == expected, (
            f"Using a custom no_python compiled distance function did not produce the"
            f"expected result. Ensure no_python compiled functions can be passed. "
            f"Expected result {expected}, got {class_result}"
        )
    else:
        expected = 50.0
        assert class_result.trace() == expected, (
            f"Using a custom NumbaDistance did not produce the expected result. Ensure"
            f"custom NumbaDistances can be passed. Expected result {expected}, got "
            f"{class_result.trace()}"
        )

        assert standalone_dist_result.trace() == expected, (
            f"Using a custom no_python compiled distance function did not produce the"
            f"expected result. Ensure no_python compiled functions can be passed."
            f"Expected result {expected}, got {class_result.trace()}"
        )


def _test_incorrect_parameters(distance_func: Callable):
    """Test to ensure correct errors thrown."""
    numpy_x = create_test_distance_numpy(10, 10)
    numpy_y = create_test_distance_numpy(10, 10, random_state=2)

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

    with pytest.raises(ValueError):  # Invalid dimensions x
        distance_func(numpy_4d, numpy_y, metric="euclidean")

    with pytest.raises(ValueError):  # Invalid dimensions y
        distance_func(numpy_x, numpy_4d, metric="euclidean")

    with pytest.raises(ValueError):  # Object that doesn't inherit NumbaDistance
        distance_func(numpy_x, numpy_y, metric=_InvalidTestClass())
