import numpy as np
import pytest
from scipy.stats import chi2, kstest

from sktime.detection._skchange.change_detectors import CROPS, PELT
from sktime.detection._skchange.change_scores import RankScore
from sktime.detection._skchange.costs._rank_cost import RankCost
from sktime.detection._skchange.datasets import (
    generate_piecewise_normal_data,
)


def test_rank_cost_single_variable_full_span():
    # Single variable, no change
    X = np.arange(10).reshape(-1, 1)
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0])
    ends = np.array([10])
    costs = cost._evaluate_optim_param(starts, ends)
    # Cost evaluated over entire interval should be zero:
    assert costs.shape == (1, 1)
    assert costs[0, 0] == pytest.approx(0.0)


def test_rank_cost_single_variable_with_change():
    # Single variable, clear change in distribution
    X = np.concatenate([np.random.rand(5), np.random.rand(5) * 10]).reshape(-1, 1)
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0, 0, 5])
    ends = np.array([10, 5, 10])
    costs = cost._evaluate_optim_param(starts, ends)
    # Both segments should have negative costs, but different values
    assert costs.shape == (3, 1)
    assert costs[0, 0] - (costs[1, 0] + costs[2, 0]) > 0


def test_rank_cost_multivariate_full_span():
    # Multivariate, over full span, zero cost.
    X = np.tile(np.arange(10), (2, 1)).T
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0])
    ends = np.array([10])
    costs = cost._evaluate_optim_param(starts, ends)
    assert costs.shape == (1, 1)
    assert costs[0, 0] == pytest.approx(0.0)


def test_rank_cost_multivariate_with_change():
    # Multivariate, change in one variable.
    # Average ranks the same in each interval [0, 5)
    # and [5, 10), thus they get the same cost.
    np.random.seed(423)
    X = np.zeros((10, 2))
    X[:5, 0] = 1 * np.random.rand(5)
    X[5:, 0] = 10 * np.random.rand(5)
    X[:, 1] = np.arange(10)
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0, 5])
    ends = np.array([5, 10])
    costs = cost._evaluate_optim_param(starts, ends)
    assert costs.shape == (2, 1)
    assert costs[0, 0] == pytest.approx(costs[1, 0])


def test_rank_cost_multivariate_change_both_vars():
    # Multivariate, change in both variables.
    # Same here, same average ranks in each interval [0, 5)
    # and [5, 10), thus they get the same cost.
    X = np.zeros((10, 2))
    X[:5, 0] = 1
    X[5:, 0] = 10
    X[:5, 1] = 2
    X[5:, 1] = 20
    cost = RankCost()
    cost.fit(X)

    # Check that the centered ranks are as expected
    # when there are ties:
    np.testing.assert_array_equal(
        cost._centered_data_ranks,
        np.array(
            [
                [-2.5, -2.5],
                [-2.5, -2.5],
                [-2.5, -2.5],
                [-2.5, -2.5],
                [-2.5, -2.5],
                [2.5, 2.5],
                [2.5, 2.5],
                [2.5, 2.5],
                [2.5, 2.5],
                [2.5, 2.5],
            ]
        ),
    )

    starts = np.array([0, 5])
    ends = np.array([5, 10])
    costs = cost._evaluate_optim_param(starts, ends)
    assert costs.shape == (2, 1)
    assert costs[0, 0] == pytest.approx(costs[1, 0])


def test_rank_cost_min_size_property():
    cost = RankCost()
    assert cost.min_size == 2


def test_rank_cost_model_size():
    cost = RankCost()
    assert cost.get_model_size(3) == 6


def test_rank_cost_on_changing_mv_normal():
    lengths = [100, 150, 125]
    changing_mv_gaussian_data = generate_piecewise_normal_data(
        means=[0, 5, 10],
        variances=[1, 3, 2],
        lengths=lengths,
        n_variables=5,
        seed=612,
    )

    expected_change_points = np.cumsum(lengths)[:-1]

    cost = RankCost()
    pruning_pelt_cpd = PELT(cost=cost, min_segment_length=5, prune=True)
    pruning_pelt_cpd.fit(changing_mv_gaussian_data)
    pruning_pelt_change_points = pruning_pelt_cpd.predict(changing_mv_gaussian_data)

    no_prune_pelt_cpd = PELT(cost=cost, min_segment_length=5, prune=False)
    no_prune_pelt_cpd.fit(changing_mv_gaussian_data)
    no_prune_pelt_change_points = no_prune_pelt_cpd.predict(changing_mv_gaussian_data)

    assert len(no_prune_pelt_change_points) == len(
        pruning_pelt_change_points
    ), "Pruned and unpruned PELT change points do not match."
    assert (
        no_prune_pelt_change_points["ilocs"] == pruning_pelt_change_points["ilocs"]
    ).all(), "Pruned and unpruned PELT change points do not match."

    crops_detector = CROPS(
        cost=cost,
        min_segment_length=2,
        min_penalty=1.0e1,
        max_penalty=1.0e3,
        selection_method="elbow",
    )
    crops_detector.fit(changing_mv_gaussian_data)

    pred_crops_change_points = crops_detector.predict(changing_mv_gaussian_data)
    assert len(pred_crops_change_points) == len(
        expected_change_points
    ), "CROPS change points do not match expected change points"
    assert (
        np.abs(pred_crops_change_points.ilocs - expected_change_points) < 5
    ).all(), "CROPS change points do not match expected change points"


def test_change_score_distribution():
    # TODO: Test distribution of change score on multivariate Gaussian data:
    # n = 200 samples, with cut point at n/8, n/2, 7*n/8.
    np.random.seed(510)
    cost = RankCost()
    # rank_change_score = to_change_score(cost)

    n_distribution_samples = 500
    data_length = 200

    cut_points = [data_length // 8, data_length // 2, 7 * data_length // 8]
    change_score_samples = np.zeros((n_distribution_samples, len(cut_points)))
    change_score_cuts = [
        np.array([[0, cut_point], [cut_point, data_length]]) for cut_point in cut_points
    ]

    n_variables = 10

    for i in range(n_distribution_samples):
        sample = generate_piecewise_normal_data(
            n_samples=data_length,
            n_variables=n_variables,
            means=[0],
            variances=[1],
            lengths=[data_length],
            seed=61 + i,
        )

        cost.fit(sample)
        for j, change_score_cut in enumerate(change_score_cuts):
            change_score = -cost.evaluate(change_score_cut).sum()
            change_score_samples[i, j] = change_score

    # Use Kolmogorov-Smirnov test to compare to chi2 distribution:
    chi2_at_n_variables_df = chi2(df=n_variables)
    for j, cut_point in enumerate(cut_points):
        res = kstest(change_score_samples[:, j], chi2_at_n_variables_df.cdf)
        assert (
            res.pvalue > 0.01
        ), f"KS test failed for cut at {cut_point}: p={res.pvalue}"


def test_split_RankCost_relation(
    # full_segment_cuts, split_cuts_list, rank_cost: RankCost
):
    np.random.seed(521)
    lengths = [50, 80, 40]
    changing_mv_gaussian_data = generate_piecewise_normal_data(
        means=[0, 5.0, 2.5], variances=[1, 1.5, 0.9], lengths=lengths, n_variables=3
    )

    rank_cost = RankCost().fit(changing_mv_gaussian_data)
    rank_score = RankScore().fit(changing_mv_gaussian_data)

    start_index = 0
    end_index = changing_mv_gaussian_data.shape[0]
    # end_index = 50

    splits = np.arange(start_index + 2, end_index - 1)
    full_segment_cuts = np.array([[start_index, end_index] for _ in splits])
    split_cuts_list = [
        np.array([[start_index, split], [split, end_index]]) for split in splits
    ]
    change_score_cuts = np.array([[start_index, split, end_index] for split in splits])

    full_segment_costs = np.array(
        [
            rank_cost.evaluate(full_segment_cut)[0, 0]
            for full_segment_cut in full_segment_cuts
        ]
    )
    split_segment_costs = np.array(
        [
            [
                rank_cost.evaluate(split_cuts[0])[0, 0],
                rank_cost.evaluate(split_cuts[1])[0, 0],
            ]
            for split_cuts in split_cuts_list
        ]
    )
    change_score_values = rank_score.evaluate(change_score_cuts)[:, 0]

    # Record the difference in full-segment vs. sum(split-segment) costs:
    actual_full_minus_split_costs = full_segment_costs - split_segment_costs.sum(axis=1)

    split_segment_lengths = np.array(
        [
            [
                split_cuts[0][1] - split_cuts[0][0],
                split_cuts[1][1] - split_cuts[1][0],
            ]
            for split_cuts in split_cuts_list
        ]
    )
    full_segment_lengths = np.array(
        [
            full_segment_cut[1] - full_segment_cut[0]
            for full_segment_cut in full_segment_cuts
        ]
    )[:, None]

    # First part reconstructing the full interval cost from the split costs:
    reconstructed_full_segment_costs = np.sum(
        (split_segment_lengths / full_segment_lengths) * split_segment_costs, axis=1
    )

    # Now add weighted inner product:
    cross_term_inner_product_weights = 2 * (
        np.prod(split_segment_lengths, axis=1)[:, None] / full_segment_lengths
    )
    cross_term_inner_products = np.zeros((full_segment_cuts.shape[0], 1))

    normalization_constant = 4.0 / np.square(rank_cost.n_samples)
    pred_full_minus_split_costs = np.zeros(full_segment_cuts.shape[0])

    for i, split_cuts in enumerate(split_cuts_list):
        start_1, split_end = split_cuts[0]
        split_start, end_2 = split_cuts[1]

        avg_ranks_pre_split = np.mean(
            rank_cost._centered_data_ranks[start_1:split_end, :], axis=0
        )
        avg_ranks_post_split = np.mean(
            rank_cost._centered_data_ranks[split_start:end_2, :], axis=0
        )

        cross_term_inner_products[i, 0] = normalization_constant * (
            avg_ranks_pre_split.T @ rank_cost._pinv_rank_cov @ avg_ranks_post_split
        )

        pre_split_length = split_end - start_1
        post_split_length = end_2 - split_start
        full_segment_length = end_2 - start_1
        diff_avg_ranks = avg_ranks_pre_split - avg_ranks_post_split
        pred_full_minus_split_costs[i] = normalization_constant * (
            (pre_split_length * post_split_length)
            / full_segment_length
            * (diff_avg_ranks.T @ rank_cost._pinv_rank_cov @ diff_avg_ranks)
        )

    cross_term_contributions = -(
        cross_term_inner_product_weights * cross_term_inner_products
    )[:, 0]
    reconstructed_full_segment_costs += cross_term_contributions

    # The full segment cost should be at least the split segment cost
    assert (full_segment_costs - reconstructed_full_segment_costs >= -1e-10).all()
    assert (actual_full_minus_split_costs - pred_full_minus_split_costs < 1e-10).all()
    assert (change_score_values - pred_full_minus_split_costs < 1e-10).all()
