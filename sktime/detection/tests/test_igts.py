"""Tests for igts module."""

import numpy as np
import pytest

from sktime.detection.igts import IGTS, InformationGainSegmentation, entropy
from sktime.tests.test_switch import run_test_for_class


@pytest.fixture
def multivariate_mean_shift():
    """Generate simple mean shift time series."""
    x = np.concatenate(tuple(np.ones(5) * i**2 for i in range(4)))
    return np.vstack((x, x.max() - x)).T


@pytest.mark.skipif(
    not run_test_for_class(entropy),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_entropy():
    """Test entropy function."""
    assert entropy(np.ones((10, 1))) == 0.0


@pytest.mark.skipif(
    not run_test_for_class(IGTS),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_igts_identity():
    """Test identity segmentation."""
    X = np.random.random(12).reshape(4, 3)
    id_change_points = IGTS().identity(X)
    assert id_change_points == [0, 4]


@pytest.mark.skipif(
    not run_test_for_class(IGTS),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_igts_get_candidates():
    """Test get_candidates function."""
    candidates = IGTS(step=2).get_candidates(n_samples=10, change_points=[0, 4, 6])
    assert candidates == [2, 8]


@pytest.mark.skipif(
    not run_test_for_class(IGTS),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_IGTS_find_change_points(multivariate_mean_shift):
    """Test the IGTS core estimator."""
    igts = IGTS(k_max=3, step=1)
    pred = igts.find_change_points(multivariate_mean_shift)
    assert isinstance(pred, list)
    assert len(pred) == 5


@pytest.mark.skipif(
    not run_test_for_class(InformationGainSegmentation),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_InformationGainSegmentation(multivariate_mean_shift):
    """Test the InformationGainSegmentation."""
    igts = InformationGainSegmentation(k_max=3, step=1)
    assert igts.get_params() == {"k_max": 3, "step": 1}
    pred = igts.fit_predict(multivariate_mean_shift)
    assert np.array_equal(
        pred,
        np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=np.int32
        ),
    )
    assert igts.change_points_ == [0, 5, 10, 15, 20]
    assert len(igts.intermediate_results_) == 3


@pytest.mark.skipif(
    not run_test_for_class(IGTS),
    reason="skip test if required soft dependency for hmmlearn not available",
)
@pytest.mark.parametrize("step", [1, 3])
def test_igts_candidate_scores_match_information_gain_score(step):
    """Test that the vectorized candidate scores equal the per-candidate scores.

    The search in ``find_change_points`` scores all candidates at once from
    cumulative sums; the scores must agree with ``information_gain_score``
    of the segmentation extended by each candidate.
    """
    from sktime.detection.igts import _information_gain_scores_of_candidates

    rng = np.random.default_rng(42)
    X = rng.random((50, 3))
    n_samples, n_series = X.shape
    cumsums = np.zeros((n_samples + 1, n_series))
    cumsums[1:] = np.cumsum(X, axis=0)

    igts = IGTS(k_max=4, step=step)
    change_points = [0, 17, 31, n_samples]
    candidates = igts.get_candidates(n_samples, change_points)

    scores = _information_gain_scores_of_candidates(
        cumsums, entropy(X), change_points, candidates
    )
    expected = [
        igts.information_gain_score(X, sorted({c, *change_points})) for c in candidates
    ]
    np.testing.assert_allclose(scores, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(
    not run_test_for_class(IGTS),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_igts_find_change_points_matches_exhaustive_search():
    """Test find_change_points against an exhaustive greedy search."""
    rng = np.random.default_rng(0)
    X = rng.random((80, 4))
    X[20:45] += 2.0
    X[60:] *= 3.0

    igts = IGTS(k_max=3, step=2)
    change_points = igts.find_change_points(X)

    # greedy top-down search, scoring every candidate separately
    expected = igts.identity(X)
    for k in range(igts.k_max):
        candidates = igts.get_candidates(X.shape[0], expected)
        scores = [
            igts.information_gain_score(X, sorted({c, *expected})) for c in candidates
        ]
        expected.append(candidates[int(np.argmax(scores))])
        expected.sort()
        np.testing.assert_allclose(
            igts.intermediate_results_[k].score, max(scores), rtol=1e-10
        )

    assert change_points == expected
