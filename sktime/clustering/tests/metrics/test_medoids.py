# -*- coding: utf-8 -*-
"""Tests for medoids."""
import numpy as np

from sktime.clustering.metrics.medoids import medoids
from sktime.distances.tests._utils import create_test_distance_numpy


def test_medoids():
    """Test medoids."""
    X = create_test_distance_numpy(10, 3, 3, random_state=2)
    test_medoids = medoids(X).tolist()
    assert np.array_equal(
        test_medoids,
        [
            [-0.16783866926192373, 0.3056703897868587, 0.023985295934094188],
            [-0.41456764450738937, 0.04385510920416571, 0.5001829432753475],
            [-0.19054625875767497, -0.18783471153949682, -0.03723538144699049],
        ],
    )
