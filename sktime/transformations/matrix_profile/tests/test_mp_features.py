"""Tests for MatrixProfileFeatures and its STOMP implementation."""

import numpy as np
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.matrix_profile import MatrixProfileFeatures
from sktime.transformations.matrix_profile._mp_features import (
    _calculate_distance_profile,
    _minimum_distance,
    _stomp_self,
)


def _z_normalized_distance(a, b):
    """Z-normalized Euclidean distance between two subsequences, by definition."""
    a = (a - np.mean(a)) / np.std(a)
    b = (b - np.mean(b)) / np.std(b)
    return np.sqrt(np.sum((a - b) ** 2))


def _matrix_profile_brute_force(ts, m):
    """Matrix profile by brute force, excluding trivial matches."""
    n_subs = len(ts) - m + 1
    mp = np.full(n_subs, np.inf)
    for i in range(n_subs):
        for j in range(n_subs):
            if j < i - m / 2 or j > i + m / 2:
                d = _z_normalized_distance(ts[i : i + m], ts[j : j + m])
                mp[i] = min(mp[i], d)
    return mp


@pytest.mark.skipif(
    not run_test_for_class(MatrixProfileFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n, m", [(50, 5), (120, 10), (200, 25)])
def test_stomp_self_matches_brute_force(n, m):
    """STOMP should agree with the brute force matrix profile."""
    rng = np.random.default_rng(42)
    ts = rng.standard_normal(n)

    mp = _stomp_self(ts, m)
    expected = _matrix_profile_brute_force(ts, m)

    np.testing.assert_allclose(mp, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    not run_test_for_class(MatrixProfileFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_stomp_self_motif():
    """A repeated pattern should have a near-zero matrix profile at its occurrences."""
    rng = np.random.default_rng(0)
    m = 20
    pattern = np.sin(np.linspace(0, 4 * np.pi, m))
    ts = rng.standard_normal(200)
    ts[30 : 30 + m] = pattern
    ts[140 : 140 + m] = pattern

    mp = _stomp_self(ts, m)

    assert mp[30] < 1e-8
    assert mp[140] < 1e-8
    # the nearest neighbours of the motif occurrences are each other
    assert np.argmin(mp) in (30, 140)


@pytest.mark.skipif(
    not run_test_for_class(MatrixProfileFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_minimum_distance_exclusion_zone():
    """_minimum_distance ignores the exclusion zone around i, and nan distances."""
    m = 4
    dp = np.array([5.0, 1.0, 2.0, 0.5, 3.0, np.nan, 0.1, 9.0])
    n = len(dp)

    # i = 3: indices with k < 1 or k > 5 are candidates, i.e., 0, 6, 7
    mp, ip = _minimum_distance(np.full(n, np.inf), np.zeros(n), dp, 3, m, n)
    assert mp[3] == 0.1
    assert ip[3] == 6

    # i = 6: candidates are k < 4, i.e., 0, 1, 2, 3
    mp, ip = _minimum_distance(np.full(n, np.inf), np.zeros(n), dp, 6, m, n)
    assert mp[6] == 0.5
    assert ip[6] == 3

    # no candidate smaller than inf: inf and index -1
    dp_inf = np.full(n, np.inf)
    mp, ip = _minimum_distance(np.full(n, np.inf), np.zeros(n), dp_inf, 3, m, n)
    assert mp[3] == np.inf
    assert ip[3] == -1


@pytest.mark.skipif(
    not run_test_for_class(MatrixProfileFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_calculate_distance_profile_matches_definition():
    """Distance profile from dot products matches the z-normalized distance."""
    rng = np.random.default_rng(1)
    m = 8
    ts = rng.standard_normal(60)
    q = ts[10 : 10 + m]
    n_subs = len(ts) - m + 1

    dot_prod = np.array([np.dot(q, ts[j : j + m]) for j in range(n_subs)])
    t_mean = np.array([np.mean(ts[j : j + m]) for j in range(n_subs)])
    t_std = np.array([np.std(ts[j : j + m]) for j in range(n_subs)])

    dp = _calculate_distance_profile(
        dot_prod, np.mean(q), np.std(q), t_mean, t_std, m, n_subs
    )
    expected = [_z_normalized_distance(q, ts[j : j + m]) for j in range(n_subs)]

    np.testing.assert_allclose(dp, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    not run_test_for_class(MatrixProfileFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_matrix_profile_features_output():
    """MatrixProfileFeatures returns one row per instance and n - m + 1 columns."""
    rng = np.random.default_rng(2)
    n_instances, n_timepoints, m = 4, 40, 6
    X = rng.standard_normal((n_instances, 1, n_timepoints))

    Xt = MatrixProfileFeatures(m=m).fit_transform(X)

    assert Xt.shape == (n_instances, n_timepoints - m + 1)
    for i in range(n_instances):
        np.testing.assert_allclose(
            Xt.iloc[i].to_numpy(),
            _matrix_profile_brute_force(X[i, 0], m),
            rtol=1e-6,
            atol=1e-6,
        )
