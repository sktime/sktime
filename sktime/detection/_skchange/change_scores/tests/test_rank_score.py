import numpy as np
from scipy.stats import chi2, kstest

from sktime.detection._skchange.change_scores._rank_score import direct_rank_score
from sktime.detection._skchange.costs._rank_cost import _compute_ranks_and_pinv_cdf_cov
from sktime.detection._skchange.datasets import (
    generate_piecewise_normal_data,
)
from sktime.detection._skchange.utils.numba import njit


@njit
def _naiive_rank_score(
    change_cuts: np.ndarray,
    centered_data_ranks: np.ndarray,
    pinv_rank_cov: np.ndarray,
) -> np.ndarray:
    """Compute the rank cost for segments.

    This function computes the rank cost for each segment defined by the
    start and end indices. The rank cost is based on the mean ranks of the
    data within each segment and the pseudo-inverse of the rank covariance
    matrix.

    Parameters
    ----------
    segment_starts : np.ndarray
        The start indices of the segments.
    segment_ends : np.ndarray
        The end indices of the segments.
    centered_data_ranks : np.ndarray
        The centered data ranks.
    pinv_rank_cov : np.ndarray
        The pseudo-inverse of the rank covariance matrix.

    Returns
    -------
    np.ndarray
        The rank costs for each segment.
    """
    n_variables = centered_data_ranks.shape[1]
    scores = np.zeros(change_cuts.shape[0])

    # Compute mean ranks for each segment:
    mean_segment_ranks = np.zeros(n_variables)

    for i, cut in enumerate(change_cuts):
        # Unpack cut row into start, split, end:
        segment_start, segment_split, segment_end = cut

        rank_score = 0.0
        full_segment_length = segment_end - segment_start
        pre_split_length = segment_split - segment_start
        post_split_length = segment_end - segment_split

        normalization_constant = 4.0 / np.square(full_segment_length)

        # First add score from first part: [start, split):
        for var in range(n_variables):
            mean_segment_ranks[var] = np.mean(
                centered_data_ranks[segment_start:segment_split, var]
            )
        rank_score += pre_split_length * (
            mean_segment_ranks.T @ pinv_rank_cov @ mean_segment_ranks
        )

        # Then add score from second part: [split, end):
        for var in range(n_variables):
            mean_segment_ranks[var] = np.mean(
                centered_data_ranks[segment_split:segment_end, var]
            )
        rank_score += post_split_length * (
            mean_segment_ranks.T @ pinv_rank_cov @ mean_segment_ranks
        )

        scores[i] = rank_score * normalization_constant

    return scores


def test_naiive_and_direct_rank_score_equivalence_full_span():
    np.random.seed(42)
    n_samples = 10
    n_variables = 3
    X = np.random.randn(n_samples, n_variables)

    # Compute ranks and pinv cov as in _rank_cost
    centered_data_ranks, pinv_rank_cov = _compute_ranks_and_pinv_cdf_cov(X)

    # Make segmentations spanning all the samples:
    segment_splits = np.arange(1, n_samples)  # [1, 2, ..., n_samples-1]
    segment_starts = np.repeat(0, n_samples - 1)
    segment_ends = np.repeat(n_samples, n_samples - 1)
    cuts = np.column_stack((segment_starts, segment_splits, segment_ends))

    naiive_scores = _naiive_rank_score(cuts, centered_data_ranks, pinv_rank_cov)
    direct_scores = direct_rank_score(cuts, centered_data_ranks, pinv_rank_cov)

    np.testing.assert_allclose(naiive_scores, direct_scores, rtol=1e-10, atol=1e-12)


def test_naiive_and_direct_rank_score_equivalence_with_repeats_full_span():
    np.random.seed(123)
    n_samples = 10

    # Create X with repeated integer values:
    X = np.array(
        [
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 8],
            [7, 8, 9],
            [7, 8, 9],
            [1, 2, 3],
            [4, 5, 2],
            [7, 8, 9],
            [1, 4, 3],
        ]
    )

    centered_data_ranks, pinv_rank_cov = _compute_ranks_and_pinv_cdf_cov(X)

    segment_splits = np.arange(1, n_samples)
    segment_starts = np.repeat(0, n_samples - 1)
    segment_ends = np.repeat(n_samples, n_samples - 1)
    cuts = np.column_stack((segment_starts, segment_splits, segment_ends))

    naiive_scores = _naiive_rank_score(cuts, centered_data_ranks, pinv_rank_cov)
    direct_scores = direct_rank_score(cuts, centered_data_ranks, pinv_rank_cov)

    np.testing.assert_allclose(naiive_scores, direct_scores, rtol=1e-10, atol=1e-12)


def test_naiive_and_direct_rank_score_equivalence_middle_span():
    np.random.seed(42)
    n_samples = 12
    n_variables = 3
    X = np.random.randn(n_samples, n_variables)

    # Compute ranks and pinv cov as in _rank_cost
    start_index = 2
    end_index = n_samples - 2
    direct_centered_data_ranks, full_pinv_rank_cov = _compute_ranks_and_pinv_cdf_cov(X)
    naiive_centered_data_ranks, _ = _compute_ranks_and_pinv_cdf_cov(
        X[start_index:end_index, :]
    )

    # Make segmentations spanning all the samples:
    segment_splits = np.arange(4, n_samples - 4)  # [1, 2, ..., n_samples-1]
    segment_starts = np.repeat(start_index, len(segment_splits))
    segment_ends = np.repeat(end_index, len(segment_splits))

    direct_cuts = np.column_stack((segment_starts, segment_splits, segment_ends))
    naiive_cuts = direct_cuts - start_index  # Adjust cuts for naiive ranks

    naiive_scores = _naiive_rank_score(
        naiive_cuts, naiive_centered_data_ranks, full_pinv_rank_cov
    )
    direct_scores = direct_rank_score(
        direct_cuts, direct_centered_data_ranks, full_pinv_rank_cov
    )

    np.testing.assert_allclose(naiive_scores, direct_scores, rtol=1e-10, atol=1e-12)


def test_change_score_distribution():
    np.random.seed(500)

    n_distribution_samples = 500
    data_length = 400

    cut_points = [data_length // 8, data_length // 2, 7 * data_length // 8]
    change_score_samples = np.zeros((n_distribution_samples, len(cut_points)))
    change_score_cuts = np.array(
        [[0, cut_point, data_length] for cut_point in cut_points]
    )

    n_variables = 10

    for i in range(n_distribution_samples):
        sample = generate_piecewise_normal_data(
            n_samples=data_length,
            n_variables=n_variables,
            means=[0],
            variances=[1],
            lengths=[data_length],
            seed=41 + i,  # Different seed for each sample
        )
        X = sample.values
        centered_data_ranks, pinv_rank_cov = _compute_ranks_and_pinv_cdf_cov(X)

        change_scores = direct_rank_score(
            change_score_cuts, centered_data_ranks, pinv_rank_cov
        )
        change_score_samples[i, :] = change_scores

    # Use Kolmogorov-Smirnov test to compare to chi2 distribution:
    chi2_at_n_variables_df = chi2(df=n_variables)
    for j, cut_point in enumerate(cut_points):
        res = kstest(change_score_samples[:, j], chi2_at_n_variables_df.cdf)
        assert res.pvalue > 0.05, (
            f"KS test failed at p=0.05 for cut at {cut_point}: p={res.pvalue}"
        )


def test_empty_cuts_returns_empty_scores():
    np.random.seed(42)
    n_samples = 10
    n_variables = 3
    X = np.random.randn(n_samples, n_variables)

    centered_data_ranks, pinv_rank_cov = _compute_ranks_and_pinv_cdf_cov(X)

    cuts = np.empty((0, 3), dtype=int)  # No cuts

    scores = direct_rank_score(cuts, centered_data_ranks, pinv_rank_cov)
    assert scores.shape == (0,)
