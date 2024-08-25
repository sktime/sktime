"""Test for distance path functionality."""

import numpy as np
import pytest

from sktime.distances._distance import _METRIC_INFOS, distance, distance_alignment_path
from sktime.distances.tests._utils import create_test_distance_numpy
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies


def _validate_distance_alignment_path_result(
    x: np.ndarray,
    y: np.ndarray,
    metric_str: str,
    distance_path_callable,
    kwargs_dict: dict = None,
):
    if kwargs_dict is None:
        kwargs_dict = {}

    metric_str_result = distance_alignment_path(x, y, metric=metric_str, **kwargs_dict)
    assert isinstance(metric_str_result, tuple), (
        f"The result for a distance path using the string: {metric_str} as the "
        f'"metric" parameter should return a tuple. The return type provided is of '
        f"type {type(metric_str_result)}"
    )
    assert len(metric_str_result) == 2

    assert isinstance(metric_str_result[0], list), (
        f"The result for a distance path first return using the string: {metric_str} as"
        f" the metric parameter should return a tuple. The return type provided is of "
        f"type {type(metric_str_result)}"
    )

    assert isinstance(metric_str_result[1], float), (
        f"The result for a distance path first return using the string: {metric_str} as"
        f" the metric parameter should return a tuple. The return type provided is of "
        f"type {type(metric_str_result)}"
    )

    distance_result = distance(x, y, metric=metric_str)
    path_result = distance_path_callable(x, y)

    assert distance_result == metric_str_result[1]
    assert distance_result == path_result[1]

    metric_str_result_cm = distance_alignment_path(
        x, y, metric=metric_str, return_cost_matrix=True, **kwargs_dict
    )

    assert isinstance(metric_str_result_cm, tuple), (
        f"The result for a distance path using the string: {metric_str} as the "
        f'"metric" parameter should return a tuple. The return type provided is of '
        f"type {type(metric_str_result)}"
    )
    assert len(metric_str_result_cm) == 3

    assert isinstance(metric_str_result_cm[0], list), (
        f"The result for a distance path first return using the string: {metric_str} as"
        f" the metric parameter should return a tuple. The return type provided is of "
        f"type {type(metric_str_result_cm[0])}"
    )

    assert isinstance(metric_str_result_cm[1], float), (
        f"The result for a distance path first return using the string: {metric_str} as"
        f" the metric parameter should return a tuple. The return type provided is of "
        f"type {type(metric_str_result)}"
    )
    assert metric_str_result_cm[0] == metric_str_result[0]
    assert metric_str_result_cm[1] == metric_str_result[1]


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none")
    or not run_test_module_changed("sktime.distances"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("dist", _METRIC_INFOS)
def test_distance_alignment_path(dist):
    """Test the distance paths."""
    if dist.dist_alignment_path_func is not None:
        if dist.canonical_name == "edr":
            return

        x = create_test_distance_numpy(10, 1)
        y = create_test_distance_numpy(10, 1, random_state=2)
        _validate_distance_alignment_path_result(
            x=x,
            y=y,
            metric_str=dist.canonical_name,
            distance_path_callable=dist.dist_alignment_path_func,
        )
