# -*- coding: utf-8 -*-
"""Test suite for numba pairwise distances."""

__author__ = ["chrisholder"]

from typing import Callable

import numpy as np
import pytest

from sktime.distances._distance import _METRIC_INFOS, pairwise_distance
from sktime.distances._numba_utils import _make_3d_series
from sktime.distances.base import MetricInfo, NumbaDistance
from sktime.distances.tests._shared_tests import (
    _test_incorrect_parameters,
    _test_metric_parameters,
)
from sktime.distances.tests._utils import create_test_distance_numpy


def _check_symmetric(x: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """Validate a matrix is symmetric.

    Parameters
    ----------
    x: np.ndarray (2d array)
        Matrix to test if symmetric.
    rtol: float
        The relative tolerance.
    atol: float
        The absolute tolerance.

    Returns
    -------
    bool
        True is matrix is symmetric and false if matrix not symmetric
    """
    return np.allclose(x, x.T, rtol=rtol, atol=atol)


def _validate_pairwise_result(
    x: np.ndarray,
    y: np.ndarray,
    metric_str: str,
    distance_factory: Callable,
    distance_function: Callable,
    distance_numba_class: NumbaDistance,
    kwargs_dict: dict = None,
):
    """Validate the pairwise distance gives desired result.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First time series.
    y: np.ndarray (1d, 2d or 3d array)
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
    """
    # Msm doesn't support multivariate so skip
    if len(x.shape) == 3 and x.shape[1] > 1 and metric_str == "msm":
        return
    if len(x.shape) == 2 and x.shape[0] > 1 and metric_str == "msm":
        return

    if kwargs_dict is None:
        kwargs_dict = {}
    metric_str_result = pairwise_distance(x, y, metric=metric_str, **kwargs_dict)

    expected_size = (len(x), len(y))

    assert metric_str_result.shape == expected_size, (
        f'The result for a pairwise using the string: {metric_str} as the "metric" '
        f"parameter should be of the shape {expected_size}. "
        f"Instead the result was of shape {metric_str_result.shape}."
    )

    assert isinstance(metric_str_result, np.ndarray), (
        f"The result for a pairwise using the string: {metric_str} as the "
        f'"metric" parameter should return a 2d numpy array. The return type provided '
        f"is of type {type(metric_str_result)}"
    )

    metric_factory_result = pairwise_distance(
        x, y, metric=distance_factory, **kwargs_dict
    )
    metric_numba_class_result = pairwise_distance(
        x, y, metric=distance_numba_class, **kwargs_dict
    )
    metric_dist_func_result = pairwise_distance(
        x, y, metric=distance_function, **kwargs_dict
    )

    assert isinstance(metric_factory_result, np.ndarray), (
        f"The result for a pairwise using the distance factory: "
        f'{distance_factory} as the "metric" parameter should return a 2d numpy '
        f"The return type provided is of type {type(metric_factory_result)}"
    )

    assert isinstance(metric_numba_class_result, np.ndarray), (
        f"The result for a pairwise using the NumbaDistance class: "
        f'{distance_numba_class} as the "metric" parameter should return a 2d '
        f"numpy The return type provided is of type "
        f"{type(metric_numba_class_result)}"
    )

    assert np.array_equal(metric_str_result, metric_factory_result), (
        f'The result of using the string: {metric_str} as the "metric" parameter'
        f"result does not equal the result of using the distance factory: "
        f'{distance_factory} as the "metric" parameter. These results should be '
        f"equal. The result of the pairwise calculation where metric={metric_str} "
        f"is {distance_factory}. The result of the distance calculation where "
        f"metric={distance_factory} is {metric_factory_result}."
    )

    assert np.array_equal(metric_str_result, metric_numba_class_result), (
        f'The result of using the string: {metric_str} as the "metric" parameter'
        f"result does not equal the result of using the NumbaDistance class: "
        f'{distance_numba_class} as the "metric" parameter. These results should '
        f"be equal."
        f"The result of the pairwise calculation where metric={metric_str} is "
        f"{metric_str_result}. The result of the distance calculation where "
        f"metric={distance_numba_class} is {metric_numba_class_result}."
    )

    assert np.array_equal(metric_str_result, metric_dist_func_result), (
        f'The result of using the string: {metric_str} as the "metric" parameter'
        f"result does not equal the result of using a NumbaDistance class: "
        f'{distance_function} as the "metric" parameter. These results should be '
        f"equal."
        f"The result of the pairwise calculation where metric={metric_str} is "
        f"{metric_str_result}. The result of the distance calculation where "
        f"metric={distance_function} is {metric_dist_func_result}."
    )

    metric_dist_self_func_result = pairwise_distance(
        x, metric=distance_function, **kwargs_dict
    )

    metric_str_result_to_self = pairwise_distance(
        x, x, metric=metric_str, **kwargs_dict
    )
    if metric_str != "twe" or metric_str == "lcss":
        assert metric_str_result_to_self.trace() == 0, (
            f"The pairwise distance when given two of the same time series e.g."
            f"pairwise_distance(x, x, ...), diagonal should equal 0."
            f"(np.trace(result)). Instead for the pairwise metric given where "
            f"metric={metric_str} is {metric_str_result_to_self.trace()}"
        )

    assert np.array_equal(metric_dist_self_func_result, metric_str_result_to_self)

    assert _check_symmetric(metric_str_result_to_self) is True, (
        f"The pairwise distance when given two of the same time series e.g."
        f"pairwise_distance(x, x, ...), should produce a symmetric matrix. This"
        f"means the left of the center diagonal should equal the right of the "
        f"center diagonal. This criteria is not met for the pairwise metric "
        f"{metric_str}"
    )

    _test_pw_equal_single_dists(x, y, distance_function, metric_str)


def _test_pw_equal_single_dists(
    x: np.ndarray, y: np.ndarray, distance_function: Callable, conical_name: str
) -> None:
    """Test pairwise result is equal to individual distance.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First time series
    y: np.ndarray (1d, 2d or 3d array)
        Second time series
    distance_function: Callable
        Distance function to test
    conical_name: str
        Name of the metric
    """
    if x.ndim < 2:
        return

    # use euclidean distance so this test will fail
    if conical_name == "lcss" or conical_name == "edr" or conical_name == "erp":
        return
    pw_result = pairwise_distance(x, y, metric=conical_name)

    x = _make_3d_series(x)
    y = _make_3d_series(y)

    matrix = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        curr_x = x[i]
        for j in range(len(y)):
            curr_y = y[j]
            matrix[i, j] = distance_function(curr_x, curr_y)

    assert np.allclose(matrix, pw_result)


@pytest.mark.parametrize("dist", _METRIC_INFOS)
def test_pairwise_distance(dist: MetricInfo) -> None:
    """Test pairwise distance.

    Parameters
    ----------
    dist: MetricInfo
        MetricInfo NamedTuple containing data for distance metric.
    """
    name = dist.canonical_name
    distance_numba_class = dist.dist_instance
    distance_function = dist.dist_func

    distance_factory = distance_numba_class.distance_factory

    _validate_pairwise_result(
        x=np.array([10.0]),
        y=np.array([15.0]),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
    )

    _validate_pairwise_result(
        x=create_test_distance_numpy(5),
        y=create_test_distance_numpy(5, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
    )

    _validate_pairwise_result(
        x=create_test_distance_numpy(5, 5),
        y=create_test_distance_numpy(5, 5, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
    )

    _validate_pairwise_result(
        x=create_test_distance_numpy(5, 1, 5),
        y=create_test_distance_numpy(5, 1, 5, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
    )

    _validate_pairwise_result(
        x=create_test_distance_numpy(5, 5, 5),
        y=create_test_distance_numpy(5, 5, 5, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
    )


def test_metric_parameters():
    """Ensure different parameters can be passed to pairwise."""
    _test_metric_parameters(pairwise_distance)


def test_incorrect_parameters():
    """Ensure incorrect parameters raise errors."""
    _test_incorrect_parameters(pairwise_distance)
