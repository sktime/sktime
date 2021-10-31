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
    metric_dist_func_result = distance(x, y, metric=distance_function, **kwargs_dict)

    distance_func_result = distance_function(x, y, **kwargs_dict)

    assert isinstance(metric_str_result, float), (
        f"The result for a distance using the string: {metric_str} as the "
        f'"metric" parameter should return a float. The return type provided is of '
        f"type {type(metric_str_result)}"
    )

    assert isinstance(metric_factory_result, float), (
        f"The result for a distance using the distance factory: "
        f'{distance_factory} as the "metric" parameter should return a float. '
        f"The return type provided is of type {type(metric_factory_result)}"
    )

    assert isinstance(metric_numba_class_result, float), (
        f"The result for a distance using the NumbaDistance class: "
        f'{distance_numba_class} as the "metric" parameter should return a float. '
        f"The return type provided is of type {type(metric_numba_class_result)}."
    )

    assert isinstance(distance_func_result, float), (
        f"The result for a distance using the NumbaDistance class: "
        f'{distance_numba_class} as the "metric" parameter should return a float. '
        f"The return type provided is of type {type(distance_func_result)}."
    )

    assert metric_str_result == metric_factory_result, (
        f'The result of using the string: {metric_str} as the "metric" parameter'
        f"result does not equal the result of using the distance factory: "
        f'{distance_factory} as the "metric" parameter. These results should be equal. '
        f"The result of the distance calculation where metric={metric_str} is "
        f"{metric_str_result}. The result of the distance calculation where "
        f"metric={distance_factory} is {metric_factory_result}."
    )

    assert metric_str_result == metric_numba_class_result, (
        f'The result of using the string: {metric_str} as the "metric" parameter'
        f"result does not equal the result of using a NumbaDistance class: "
        f'{distance_numba_class} as the "metric" parameter. These results should be '
        f"equal. "
        f"The result of the distance calculation where metric={metric_str} is "
        f"{metric_str_result}. The result of the distance calculation where "
        f"metric={distance_numba_class} is {metric_numba_class_result}."
    )

    assert metric_str_result == metric_dist_func_result, (
        f'The result of using the string: {metric_str} as the "metric" parameter'
        f"result does not equal the result of using a NumbaDistance class: "
        f'{distance_function} as the "metric" parameter. These results should be '
        f"equal."
        f"The result of the distance calculation where metric={metric_str} is "
        f"{metric_str_result}. The result of the distance calculation where "
        f"metric={distance_function} is {metric_dist_func_result}."
    )

    assert metric_str_result == distance_func_result, (
        f'The result of using the string: {metric_str} as the "metric" parameter'
        f"result does not equal the result of using a distance function: "
        f"{distance_function}. These results should be equal. "
        f"The result of the distance calculation where metric={metric_str} is "
        f"{metric_str_result}. Then result of the distance function "
        f"{distance_function} is {distance_func_result}."
    )

    metric_str_result_to_self = distance(x, x, metric=metric_str, **kwargs_dict)
    assert metric_str_result_to_self == 0, (
        f"The distance when given two of the same timeseries e.g."
        f"distance(x, x, ...), result should equal 0. This criteria is not met for "
        f"the metric {metric_str}. The result was {metric_str_result_to_self}"
    )

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


# def test_dist():
#     from sktime.dists_kernels.numba.distances.distance import ddtw_distance
#     from sktime.distances.elastic_cython import ddtw_distance as sktime_ddtw
#
#     dist = ddtw_distance
#
#     x_first = np.array([10.0])
#     y_first = np.array([15.0])
#
#     x_second = create_test_distance_numpy(10)
#     y_second = create_test_distance_numpy(10, random_state=2)
#
#     x_third = create_test_distance_numpy(10, 1)
#     y_third = create_test_distance_numpy(10, 1, random_state=2)
#
#     x_fourth = create_test_distance_numpy(10, 10)
#     y_fourth = create_test_distance_numpy(10, 10, random_state=2)
#
#     x_fifth = create_test_distance_numpy(10, 10, 1)
#     y_fifth = create_test_distance_numpy(10, 10, 1, random_state=2)
#
#     x_sixth = create_test_distance_numpy(10, 10, 10)
#     y_sixth = create_test_distance_numpy(10, 10, 10, random_state=2)
#
#     first = dist(x_first, y_first)
#     second = dist(x_second, y_second)
#
#     third = dist(x_third, y_third)
#     sktime_third = sktime_ddtw(x_third, y_third)
#     fourth = dist(x_fourth, y_fourth)
#     sktime_fourth = sktime_ddtw(x_fourth, y_fourth)
#
#     fifth = dist(x_fifth, y_fifth)
#     sixth = dist(x_sixth, y_sixth)
#     pass
