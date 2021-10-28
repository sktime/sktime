# -*- coding: utf-8 -*-
"""Test suite for numba distances."""

__author__ = ["chrisholder"]

from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from sktime.dists_kernels.numba.distances.base import NumbaDistance
from sktime.dists_kernels.numba.distances.distance import (
    _METRIC_INFOS,
    MetricInfo,
    distance,
)
from sktime.dists_kernels.numba.tests._expected_results import (
    _expected_distance_results,
)
from sktime.dists_kernels.numba.tests._shared_tests import (
    _test_incorrect_parameters,
    _test_metric_parameters,
)
from sktime.dists_kernels.tests._utils import create_test_distance_numpy


def _validate_distance_result(
    x: np.ndarray,
    y: np.ndarray,
    metric_str: str,
    distance_factory: Callable,
    distance_function: Callable,
    distance_numba_class: NumbaDistance,
    kwargs_dict: dict = None,
    expected_result: float = None,
):
    """Validate the distance gives desired result.

    Parameters
    ----------
    x: np.ndarray
        First timeseries.
    y: np.ndarray
        Second timeseries.
    metric_str: str
        Metric string name.
    distance_factory: Callable
        Distance factory callable
    distance_function: Callable
        Distance function callable
    distance_numba_class: Callable
        NumbaDistance class
    kwargs_dict: dict
        Extra kwargs
    expected_result:
        float that is the expected result of tests.
    """
    if kwargs_dict is None:
        kwargs_dict = {}
    metric_str_result = distance(x, y, metric=metric_str, **kwargs_dict)
    metric_factory_result = distance(x, y, metric=distance_factory, **kwargs_dict)
    metric_numba_class_result = distance(
        x, y, metric=distance_numba_class, **kwargs_dict
    )

    distance_func_result = distance_function(x, y, **kwargs_dict)

    assert isinstance(metric_str_result, float)
    assert isinstance(metric_factory_result, float)
    assert isinstance(metric_numba_class_result, float)
    assert isinstance(distance_func_result, float)

    assert metric_str_result == metric_factory_result
    assert metric_str_result == metric_numba_class_result
    assert metric_str_result == distance_func_result

    if expected_result is not None:
        assert_almost_equal(metric_str_result, expected_result, 5)


@pytest.mark.parametrize("dist", _METRIC_INFOS)
def test_distance(dist: MetricInfo) -> None:
    """Test distance.

    Parameters
    ----------
    dist: MetricInfo
        MetricInfo NamedTuple containing data for distance metric.
    """
    name = dist.canonical_name
    distance_numba_class = dist.dist_instance
    distance_function = dist.dist_func
    distance_factory = distance_numba_class.distance_factory

    _validate_distance_result(
        x=np.array([10.0]),
        y=np.array([15.0]),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][0],
    )

    _validate_distance_result(
        x=create_test_distance_numpy(10),
        y=create_test_distance_numpy(10, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][1],
    )

    _validate_distance_result(
        x=create_test_distance_numpy(10, 1),
        y=create_test_distance_numpy(10, 1, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][1],
    )

    _validate_distance_result(
        x=create_test_distance_numpy(10, 10),
        y=create_test_distance_numpy(10, 10, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][2],
    )

    _validate_distance_result(
        x=create_test_distance_numpy(10, 10, 1),
        y=create_test_distance_numpy(10, 10, 1, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][2],
    )

    _validate_distance_result(
        x=create_test_distance_numpy(10, 10, 10),
        y=create_test_distance_numpy(10, 10, 10, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][3],
    )


def test_metric_parameters():
    """Ensure different parameters can be passed to distance."""
    _test_metric_parameters(distance)


def test_incorrect_parameters():
    """Ensure incorrect parameters raise errors."""
    _test_incorrect_parameters(distance)
