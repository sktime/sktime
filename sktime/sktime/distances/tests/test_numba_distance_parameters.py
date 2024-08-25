"""Test suite for numba distances with parameters."""

from collections.abc import Callable

import numpy as np
import pytest

from sktime.distances import distance, distance_factory
from sktime.distances._distance import _METRIC_INFOS
from sktime.distances._numba_utils import to_numba_timeseries
from sktime.distances.base import MetricInfo
from sktime.distances.tests._expected_results import _expected_distance_results_params
from sktime.distances.tests._utils import create_test_distance_numpy
from sktime.tests.test_switch import run_test_for_class, run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.numba.njit import njit


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none")
    or not run_test_module_changed("sktime.distances"),
    reason="skip test if required soft dependency not available",
)
def _test_distance_params(
    param_list: list[dict], distance_func: Callable, distance_str: str
):
    x_univ = to_numba_timeseries(create_test_distance_numpy(10, 1))
    y_univ = to_numba_timeseries(create_test_distance_numpy(10, 1, random_state=2))

    x_multi = create_test_distance_numpy(10, 10)
    y_multi = create_test_distance_numpy(10, 10, random_state=2)

    test_ts = [[x_univ, y_univ], [x_multi, y_multi]]
    results_to_fill = []

    i = 0
    for param_dict in param_list:
        j = 0
        curr_results = []
        for x, y in test_ts:
            results = []
            curr_dist_fact = distance_factory(x, y, metric=distance_str, **param_dict)
            results.append(distance_func(x, y, **param_dict))
            results.append(distance(x, y, metric=distance_str, **param_dict))
            results.append(curr_dist_fact(x, y))
            if distance_str in _expected_distance_results_params:
                if _expected_distance_results_params[distance_str][i][j] is not None:
                    for result in results:
                        assert result == pytest.approx(
                            _expected_distance_results_params[distance_str][i][j]
                        )
            curr_results.append(results[0])
            j += 1
        i += 1
        results_to_fill.append(curr_results)


BASIC_BOUNDING_PARAMS = [
    {"window": 0.2},
    {"itakura_max_slope": 0.5},
    {"bounding_matrix": np.zeros((10, 10))},
]


@njit(cache=True)
def _test_derivative(q: np.ndarray):
    return q


DIST_PARAMS = {
    "dtw": BASIC_BOUNDING_PARAMS,
    "erp": BASIC_BOUNDING_PARAMS + [{"g": 0.5}],
    "edr": BASIC_BOUNDING_PARAMS + [{"epsilon": 0.5}],
    "lcss": BASIC_BOUNDING_PARAMS + [{"epsilon": 0.5}],
    "ddtw": BASIC_BOUNDING_PARAMS + [{"compute_derivative": _test_derivative}],
    "wdtw": BASIC_BOUNDING_PARAMS + [{"g": 0.5}],
    "wddtw": BASIC_BOUNDING_PARAMS
    + [{"compute_derivative": _test_derivative}]
    + [{"g": 0.5}],
    "twe": BASIC_BOUNDING_PARAMS + [{"lmbda": 0.5}, {"nu": 0.9}, {"p": 4}],
}


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none")
    or not run_test_module_changed("sktime.distances"),  # noqa: E501
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("dist", _METRIC_INFOS)
def test_distance_params(dist: MetricInfo):
    """Test parametisation of distance callables."""
    # skip test if distance function/class have not changed
    if not run_test_for_class([dist.dist_func, dist.dist_instance.__class__]):
        return None

    if dist.canonical_name in DIST_PARAMS:
        _test_distance_params(
            DIST_PARAMS[dist.canonical_name],
            dist.dist_func,
            dist.canonical_name,
        )
