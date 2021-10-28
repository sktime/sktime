# -*- coding: utf-8 -*-
"""Test suite for numba pairwise distances."""

__author__ = ["chrisholder"]

from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from sktime.dists_kernels.numba.distances.base import NumbaDistance
from sktime.dists_kernels.numba.distances.distance import _METRIC_INFOS, MetricInfo
from sktime.dists_kernels.numba.distances.pairwise_distance import pairwise_distance
from sktime.dists_kernels.numba.tests._expected_results import (
    _expected_distance_results,
)
from sktime.dists_kernels.numba.tests._shared_tests import (
    _test_incorrect_parameters,
    _test_metric_parameters,
)
from sktime.dists_kernels.tests._utils import create_test_distance_numpy


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
    expected_result: float = None,
):
    """Validate the pairwise distance gives desired result.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
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
    metric_str_result = pairwise_distance(x, y, metric=metric_str, **kwargs_dict)
    metric_factory_result = pairwise_distance(
        x, y, metric=distance_factory, **kwargs_dict
    )
    metric_numba_class_result = pairwise_distance(
        x, y, metric=distance_numba_class, **kwargs_dict
    )
    metric_dist_func_result = pairwise_distance(
        x, y, metric=distance_function, **kwargs_dict
    )

    assert isinstance(metric_str_result, np.ndarray), (
        f"The result for a pairwise using the string: {metric_str} as the "
        f'"metric" parameter should return a 2d numpy array. It currently does not.'
    )

    assert isinstance(metric_factory_result, np.ndarray), (
        f"The result for a pairwise using the distance factory: "
        f'{distance_factory} as the "metric" parameter should return a 2d numpy '
        f"array. It currently does not."
    )

    assert isinstance(metric_numba_class_result, np.ndarray), (
        f"The result for a pairwise using the NumbaDistance class: "
        f'{distance_numba_class} as the "metric" parameter should return a 2d '
        f"numpy. It currently does not."
    )

    assert np.array_equal(metric_str_result, metric_factory_result), (
        f'The result of using the string: {metric_str} as the "metric" parameter'
        f"result does not equal the result of using the distance factory: "
        f'{distance_factory} as the "metric" parameter. These results should be equal.'
    )

    assert np.array_equal(metric_str_result, metric_numba_class_result), (
        f'The result of using the string: {metric_str} as the "metric" parameter'
        f"result does not equal the result of using the NumbaDistance class: "
        f'{distance_numba_class} as the "metric" parameter. These results should '
        f"be equal."
    )

    assert np.array_equal(metric_str_result, metric_dist_func_result), (
        f'The result of using the string: {metric_str} as the "metric" parameter'
        f"result does not equal the result of using a NumbaDistance class: "
        f'{distance_function} as the "metric" parameter. These results should be '
        f"equal."
    )

    metric_str_result_to_self = pairwise_distance(
        x, x, metric=metric_str, **kwargs_dict
    )
    assert metric_str_result_to_self.trace() == 0, (
        f"The pairwise distance when given two of the same timeseries e.g."
        f"pairwise_distance(x, x, ...), diagonal should equal 0."
        f"(np.trace(result)). This criteria is not met for the pairwise metric "
        f"{metric_str}"
    )

    assert _check_symmetric(metric_str_result_to_self) is True, (
        f"The pairwise distance when given two of the same timeseries e.g."
        f"pairwise_distance(x, x, ...), should produce a symmetric matrix. This"
        f"means the left of the center diagonal should equal the right of the center"
        f"diagonal. This criteria is not met for the pairwise metric {metric_str}"
    )

    if expected_result is not None:
        assert_almost_equal(metric_str_result.trace(), expected_result, 5)


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
        expected_result=_expected_distance_results[name][0],
    )

    _validate_pairwise_result(
        x=create_test_distance_numpy(10),
        y=create_test_distance_numpy(10, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][1],
    )

    _validate_pairwise_result(
        x=create_test_distance_numpy(10, 1),
        y=create_test_distance_numpy(10, 1, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][1],
    )

    _validate_pairwise_result(
        x=create_test_distance_numpy(10, 10),
        y=create_test_distance_numpy(10, 10, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][2],
    )

    _validate_pairwise_result(
        x=create_test_distance_numpy(10, 10, 1),
        y=create_test_distance_numpy(10, 10, 1, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][2],
    )

    _validate_pairwise_result(
        x=create_test_distance_numpy(10, 10, 10),
        y=create_test_distance_numpy(10, 10, 10, random_state=2),
        metric_str=name,
        distance_factory=distance_factory,
        distance_function=distance_function,
        distance_numba_class=distance_numba_class,
        expected_result=_expected_distance_results[name][3],
    )


def test_metric_parameters():
    """Ensure different parameters can be passed to pairwise."""
    _test_metric_parameters(pairwise_distance)


def test_incorrect_parameters():
    """Ensure incorrect parameters raise errors."""
    _test_incorrect_parameters(pairwise_distance)
