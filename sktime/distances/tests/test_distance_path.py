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

def test_distance_path():
    from sktime.distances._distance import msm_path

    x= create_test_distance_numpy(10, 1)
    y= create_test_distance_numpy(10, 1, random_state=2)
    callable_path = msm_path
    metric = 'msm'

    path, dist = distance_path(x, y, metric)
    path2, dist2 = callable_path(x, y)
    joe = ''
