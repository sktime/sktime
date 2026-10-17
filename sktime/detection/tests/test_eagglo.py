"""Tests for E-Agglo (agglomerative clustering algorithm)."""

__author__ = ["KatieBuc"]

import numpy as np
import pandas as pd
import pytest

from sktime.detection.eagglo import EAgglo
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(EAgglo),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_default_params_univariate():
    """Test univariate data and default parameters.

    These numbers are generated from the original implementation in R,
    with the following code:

    set.seed(1234) X <- c(rnorm(15, mean = -6), 0, rnorm(16, mean = 6)) X <-
    as.matrix(X[c(1,2,17,18)]) ret = e.agglo(X)
    """
    X = pd.DataFrame([-7.207066, -5.722571, 5.889715, 5.488990])

    cluster_expected = [0, 0, 1, 1]
    fit_expected = [104.77424, 134.51387, 186.92586, -31.15431]

    model = EAgglo()
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster_
    fit_actual = fitted_model.gof_

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


@pytest.mark.skipif(
    not run_test_for_class(EAgglo),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_other_params_univariate():
    """Test univariate data with alternative starting clusters."""
    X = pd.DataFrame([-7.207066, -5.722571, 5.889715, 5.488990])

    cluster_expected = [0, 0, 1, 1]
    fit_expected = [1182.754, 1772.526, -295.421]

    model = EAgglo(member=np.array([0, 0, 1, 2]), alpha=2)
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster_
    fit_actual = fitted_model.gof_

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


@pytest.mark.skipif(
    not run_test_for_class(EAgglo),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_default_params_multivariate():
    """Test multivariate data with default parameters.

    These numbers are generated from the original implementation
    in R, with the following code:

    set.seed(1234)
    X <- c(rnorm(15, mean = -6), 0, rnorm(16, mean = 6))
    X <- as.matrix(cbind(X[c(1,2,17,18,19)],X[c(3,4,5,6,7)]))
    ret = e.agglo(X)
    """
    X = pd.DataFrame(
        [
            [-7.207, -4.916],
            [-5.723, -8.346],
            [5.890, -5.571],
            [5.489, -5.494],
            [5.089, -6.575],
        ]
    )

    cluster_expected = [0, 0, 1, 1, 1]
    fit_expected = [118.58235, 132.67919, 156.54531, 208.61512, -33.52743]

    model = EAgglo()
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster_
    fit_actual = fitted_model.gof_

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


def test_len_penalty():
    """Test multivariate data with penalty function as string input."""
    X = pd.DataFrame(
        [
            [-7.207, -4.916],
            [-5.723, -8.346],
            [5.890, -5.571],
            [5.489, -5.494],
            [5.089, -6.575],
        ]
    )

    cluster_expected = [0, 0, 1, 1, 1]
    fit_expected = [112.58235, 127.67919, 152.54531, 205.61512, -35.52743]

    model = EAgglo(penalty="len_penalty")
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster_
    fit_actual = fitted_model.gof_

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


def test_custom_penalty():
    """Test multivariate data with functional input as penalty."""
    X = pd.DataFrame([-7.207066, -5.722571, 5.889715, 5.488990])

    cluster_expected = [0, 0, 1, 1]
    fit_expected = [105.77424, 135.84720, 188.92586, -29.15431]

    model = EAgglo(penalty=lambda x: np.mean(np.diff(np.sort(x))))
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster_
    fit_actual = fitted_model.gof_

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


@pytest.mark.skipif(
    not run_test_for_class(EAgglo),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "cluster_sizes", [[1, 1, 1, 1, 1, 1], [3, 3], [2, 1, 3], [6], [1, 2, 3]]
)
@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
def test_initial_distances(cluster_sizes, alpha):
    """Test that initial distances agree with the between-within definition.

    ``_initial_distances`` computes the distances of the initial clusters in
    blocks, which must agree with the definition, twice the mean distance
    between two clusters minus the mean distance within either cluster.
    """
    from sktime.detection.eagglo import _initial_distances, get_distance

    cluster_sizes = np.array(cluster_sizes)
    n_cluster = len(cluster_sizes)
    X = np.random.RandomState(42).normal(size=(cluster_sizes.sum(), 2))

    starts = np.concatenate(([0], np.cumsum(cluster_sizes)[:-1]))
    clusters = [X[i : i + size] for i, size in zip(starts, cluster_sizes)]
    within = [get_distance(x, x, alpha) for x in clusters]

    expected = np.array(
        [
            [
                2 * get_distance(clusters[i], clusters[j], alpha)
                - within[i]
                - within[j]
                for j in range(n_cluster)
            ]
            for i in range(n_cluster)
        ]
    )

    # the result must not depend on the size of the blocks the distances
    # are computed in, including blocks of a single cluster
    for max_block in [1, 4, 2**18]:
        actual = np.empty((n_cluster, n_cluster))
        _initial_distances(X, cluster_sizes, alpha, out=actual, max_block=max_block)
        assert np.allclose(actual, expected, atol=1e-12)


@pytest.mark.skipif(
    not run_test_for_class(EAgglo),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_unsorted_member_raises():
    """Test that unsorted cluster membership raises."""
    X = pd.DataFrame(np.random.RandomState(1).normal(size=(6, 2)))

    with pytest.raises(ValueError, match="should be sorted"):
        EAgglo(member=np.array([0, 1, 0, 1, 2, 2]))._fit(X)


@pytest.mark.skipif(
    not run_test_for_class(EAgglo),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("member", [np.repeat([0, 1, 2, 3], 3), np.repeat([0, 1], 6)])
def test_relabels_non_consecutive_member(member):
    """Test that non-consecutive cluster labels give the same result as relabeled.

    Cluster labels are relabeled to consecutive numbers, so labels that are not
    consecutive must give the same clustering as the consecutive ones.
    """
    X = pd.DataFrame(np.random.RandomState(2).normal(size=(12, 2)).cumsum(axis=0))

    expected = EAgglo(member=member)._fit(X)
    actual = EAgglo(member=3 * np.asarray(member) + 2)._fit(X)

    assert np.array_equal(actual.cluster_, expected.cluster_)
    assert np.allclose(actual.gof_, expected.gof_)
