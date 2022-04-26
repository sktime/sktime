# -*- coding: utf-8 -*-
"""Test suite for numba distances."""

__author__ = ["chrisholder"]

from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from sktime.distances._distance import _METRIC_INFOS, distance
from sktime.distances.base import MetricInfo, NumbaDistance
from sktime.distances.tests._expected_results import _expected_distance_results
from sktime.distances.tests._shared_tests import (
    _test_incorrect_parameters,
    _test_metric_parameters,
)
from sktime.distances.tests._utils import create_test_distance_numpy

_ran_once = False


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
        First time series.
    y: np.ndarray
        Second time series.
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
    if expected_result is None:
        return
    if kwargs_dict is None:
        kwargs_dict = {}
    metric_str_result = distance(x, y, metric=metric_str, **kwargs_dict)

    assert isinstance(metric_str_result, float), (
        f"The result for a distance using the string: {metric_str} as the "
        f'"metric" parameter should return a float. The return type provided is of '
        f"type {type(metric_str_result)}"
    )
    global _ran_once
    _ran_once = False

    if _ran_once is False:
        _ran_once = True
        metric_factory_result = distance(x, y, metric=distance_factory, **kwargs_dict)
        metric_numba_class_result = distance(
            x, y, metric=distance_numba_class, **kwargs_dict
        )
        metric_dist_func_result = distance(
            x, y, metric=distance_function, **kwargs_dict
        )

        distance_func_result = distance_function(x, y, **kwargs_dict)

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
            f'{distance_factory} as the "metric" parameter. These results should be '
            f"equal. The result of the distance calculation where metric={metric_str} "
            f"is {metric_str_result}. The result of the distance calculation where "
            f"metric={distance_factory} is {metric_factory_result}."
        )

        assert metric_str_result == metric_numba_class_result, (
            f'The result of using the string: {metric_str} as the "metric" parameter'
            f"result does not equal the result of using a NumbaDistance class: "
            f'{distance_numba_class} as the "metric" parameter. These results should '
            f"be equal. "
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
        f"The distance when given two of the same time series e.g."
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
        x=create_test_distance_numpy(10, 10),
        y=create_test_distance_numpy(10, 10, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][2],
    )


def test_metric_parameters():
    """Ensure different parameters can be passed to distance."""
    _test_metric_parameters(distance)


def test_incorrect_parameters():
    """Ensure incorrect parameters raise errors."""
    _test_incorrect_parameters(distance)

def test_twe():
    from sktime.distances._distance import twe_distance

    x = create_test_distance_numpy(10)
    y = create_test_distance_numpy(10, random_state=2)

    s1 = np.array([[9.868, 14.54],
                   [5.235, 16.3],
                   [6.203, 16.76],
                   [7.968, 17.13],
                   [6.493, 17.58],
                   [12.466, 16.68],
                   [13.255, 14.31],
                   [11.263, 12.09],
                   [7.155, 10.53],
                   [7.377, 9.48],
                   [8.672, 8.73],
                   [7.881, 8.58]]).astype('float64')
    s2 = np.array([[9.428, 5.06],
                   [12.91, 5.78],
                   [9.365, 7.01],
                   [12.535, 7.99],
                   [10.152, 8.89],
                   [11.473, 9.58],
                   [14.701, 8.12],
                   [10.156, 6.9],
                   [8.292, 5.82],
                   [5.946, 4.96],
                   [6.339, 4.09],
                   [5.395, 4.]]).astype('float64')

    x = s1.reshape((s1.shape[1], s1.shape[0]))
    y = s2.reshape((s2.shape[1], s2.shape[0]))

    s1 = np.array([[9.868, 14.54],
                   [5.235, 16.3],
                   [6.203, 16.76],
                   [7.968, 17.13],
                   [6.493, 17.58],
                   [12.466, 16.68],
                   [13.255, 14.31],
                   [11.263, 12.09],
                   [7.155, 10.53],
                   [7.377, 9.48],
                   [8.672, 8.73],
                   [7.881, 8.58]]).astype('float64')
    s2 = np.array([[9.428, 5.06],
                   [12.91, 5.78],
                   [9.365, 7.01],
                   [12.535, 7.99],
                   [10.152, 8.89],
                   [11.473, 9.58],
                   [14.701, 8.12],
                   [10.156, 6.9],
                   [8.292, 5.82],
                   [5.946, 4.96],
                   [6.339, 4.09],
                   [5.395, 4.]]).astype('float64')



    reshape_x = s1
    reshape_y = s2

    def reshapey(x_ts):
        reshaped = [[] for i in range(x_ts.shape[0])]
        for ts in x_ts:
            curr_start = 0
            for i in range(x_ts.shape[0]):
                for j in range(curr_start, len(ts), x_ts.shape[0]):
                    reshaped[i].append(ts[j])
                curr_start += 1
        return np.array(reshaped)

    if reshape_x.ndim > 1:
        x = reshapey(x)
        y = reshapey(y)
    else:
        x = reshape_x
        y = reshape_y

    nu = .1
    lmbda = .2
    p = 2
    from sktime.distances.tests.twetemp import twed as actual_twe

    test, cm1 = twe_distance(x, y, nu=nu, lmbda=lmbda, p=p)
    test2, cm2 = actual_twe(s1, s2, nu = nu, lmbda = lmbda, p = p, fast=True)
    joe = ''
