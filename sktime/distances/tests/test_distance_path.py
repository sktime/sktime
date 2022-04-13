from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from sktime.distances.base import MetricInfo, NumbaDistance
from sktime.distances.tests._expected_results import _expected_distance_results
from sktime.distances.tests._shared_tests import (
    _test_incorrect_parameters,
    _test_metric_parameters,
)
from sktime.distances.tests._utils import create_test_distance_numpy
from sktime.distances._distance import distance_path, dtw_path, distance, _METRIC_INFOS
from tslearn.metrics import dtw_path as tslearn_path
from tslearn.metrics.dtw_variants import _return_path

def _validate_distance_path_result(
        x: np.ndarray,
        y: np.ndarray,
        metric_str: str,
        distance_instance,
        distance_path_callable,
        kwargs_dict: dict = None,
        expected_result: float = None
):
    if expected_result is None:
        return
    if kwargs_dict is None:
        kwargs_dict = {}

    pass

@pytest.mark.parametrize("dist", _METRIC_INFOS)
def test_distance_path(dist):
    if dist.dist_path_func is not None:

        x = create_test_distance_numpy(10, 1)
        y = create_test_distance_numpy(10, 1, random_state=2)

        _validate_distance_path_result(x, y, dist.canonical_name, dist.dist_instance, dist.dist_path_func)