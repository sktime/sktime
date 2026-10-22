"""Tests for medoids."""

import numpy as np
import pytest

from sktime.clustering.metrics.medoids import medoids
from sktime.dists_kernels._numba_distances.tests._utils import (
    create_test_distance_numpy,
)
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


def test_medoids_precomputed_distance():
    """Medoid is the series with the smallest sum of distances to all others."""
    rng = np.random.default_rng(0)
    n_instances = 50
    X = rng.standard_normal((n_instances, 2, 10))
    distances = rng.random((n_instances, n_instances))
    distances = distances + distances.T
    np.fill_diagonal(distances, 0)

    result = medoids(X, precomputed_pairwise_distance=distances)

    expected_index = min(
        range(n_instances),
        key=lambda j: sum(distances[k, j] for k in range(n_instances)),
    )
    np.testing.assert_array_equal(result, X[expected_index])

    # empty input is returned unchanged
    assert medoids(X[:0], precomputed_pairwise_distance=distances[:0, :0]).shape[0] == 0
