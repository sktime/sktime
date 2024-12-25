"""Tests for medoids."""

import numpy as np
import pytest

from sktime.clustering.metrics.medoids import medoids
from sktime.distances.tests._utils import create_test_distance_numpy
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(medoids),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_medoids():
    """Test medoids."""
    X = create_test_distance_numpy(10, 3, 3, random_state=2)
    test_medoids = medoids(X)
    assert np.array_equal(
        test_medoids,
        [
            [-0.3739354746469312, 0.004512625486662562, -0.439053946620171],
            [-0.07821708519231182, 0.12828522600064782, -0.4943895243848109],
            [-0.16941098301459817, -0.11809201542630064, -0.3188275062421506],
        ],
    )
