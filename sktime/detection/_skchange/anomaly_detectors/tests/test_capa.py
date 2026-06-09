"""Tests for CAPA and all available savings."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection._skchange.anomaly_detectors import CAPA
from sktime.detection._skchange.anomaly_detectors._capa import run_capa
from sktime.detection._skchange.anomaly_scores import SAVINGS, L2Saving, to_saving
from sktime.detection._skchange.base import BaseIntervalScorer
from sktime.detection._skchange.change_scores import ChangeScore
from sktime.detection._skchange.compose.penalised_score import PenalisedScore
from sktime.detection._skchange.costs import (
    COSTS,
    EmpiricalDistributionCost,
    L1Cost,
    L2Cost,
    MultivariateGaussianCost,
)
from sktime.detection._skchange.costs.base import BaseCost
from sktime.detection._skchange.costs.tests.test_all_costs import find_fixed_param_combination
from sktime.detection._skchange.datasets import generate_alternating_data
from sktime.detection._skchange.penalties import make_nonlinear_chi2_penalty
from sktime.detection._skchange.tests.test_all_interval_scorers import skip_if_no_test_data


def make_nonlinear_chi2_penalty_from_score(
    score: BaseIntervalScorer,
) -> np.ndarray:
    score.check_is_fitted()
    n = score.n_samples
    p = score.n_variables
    return make_nonlinear_chi2_penalty(score.get_model_size(p), n, p)


COSTS_AND_SAVINGS = COSTS + SAVINGS


@pytest.mark.parametrize("Saving", COSTS_AND_SAVINGS)
def test_capa_anomalies(Saving):
    """Test CAPA anomalies."""
    saving = Saving.create_test_instance()
    skip_if_no_test_data(saving)
    if isinstance(saving, BaseCost):
        if not saving.get_tag("supports_fixed_param"):
            pytest.skip(f"{type(saving).__name__} does not support fixed parameters.")
        else:
            fixed_params = find_fixed_param_combination(saving)
            saving = saving.set_params(**fixed_params)

    if isinstance(saving, BaseCost) and not saving.get_tag("supports_fixed_param"):
        pytest.skip("Skipping test for Cost without support for fixed params.")

    if isinstance(saving, EmpiricalDistributionCost):
        pytest.skip(
            "Skipping test for EmpiricalDistributionCost, as its `fixed_params`"
            "implementation fails this CAPA test currently."
        )

    n_segments = 2
    seg_len = 50
    p = 5
    df = generate_alternating_data(
        n_segments=n_segments,
        mean=20,
        segment_length=seg_len,
        p=p,
        affected_proportion=0.2,
        random_state=8,
    )

    # Cannot use costs with min_size > 1 as point saving:
    if not isinstance(saving.min_size, int) or saving.min_size > 1:
        point_saving = L1Cost(param=0.0)
    else:
        point_saving = saving

    detector = CAPA(
        segment_saving=saving,
        point_saving=point_saving,
        min_segment_length=20,
        ignore_point_anomalies=True,  # To get test coverage.
        find_affected_components=True,  # To get test coverage.
    )
    anomalies = detector.fit_predict(df)
    if isinstance(anomalies, pd.DataFrame):
        anomalies = anomalies.iloc[:, 0]
    # End point also included as a changepoint
    assert (
        len(anomalies) == 1
        and anomalies.array.left[0] == seg_len
        and anomalies.array.right[0] == 2 * seg_len
    )


@pytest.mark.xfail(strict=True, reason="CAPA with EmpiricalDistributionCost fails.")
def test_capa_anomalies_with_EmpiricalDistributionCost():
    """Test CAPA anomalies."""
    cost = EmpiricalDistributionCost()
    skip_if_no_test_data(cost)
    if isinstance(cost, BaseCost):
        if not cost.get_tag("supports_fixed_param"):
            pytest.skip(f"{type(cost).__name__} does not support fixed parameters.")
        else:
            fixed_params = find_fixed_param_combination(cost)
            cost = cost.set_params(**fixed_params)

    if isinstance(cost, BaseCost) and not cost.get_tag("supports_fixed_param"):
        pytest.skip("Skipping test for Cost without support for fixed params.")

    if isinstance(cost, EmpiricalDistributionCost):
        pytest.skip(
            "Skipping test for EmpiricalDistributionCost, as its `fixed_params`"
            "implementation fails this CAPA test currently."
        )

    n_segments = 2
    seg_len = 50
    p = 5
    df = generate_alternating_data(
        n_segments=n_segments,
        mean=20,
        segment_length=seg_len,
        p=p,
        affected_proportion=0.2,
        random_state=8,
    )

    # Cannot use costs with min_size > 1 as point saving:
    if not isinstance(cost.min_size, int) or cost.min_size > 1:
        point_saving = L1Cost(param=0.0)
    else:
        point_saving = cost

    detector = CAPA(
        segment_saving=cost,
        point_saving=point_saving,
        min_segment_length=20,
        ignore_point_anomalies=True,  # To get test coverage.
        find_affected_components=True,  # To get test coverage.
    )
    anomalies = detector.fit_predict(df)
    if isinstance(anomalies, pd.DataFrame):
        anomalies = anomalies.iloc[:, 0]
    # End point also included as a changepoint
    assert (
        len(anomalies) == 1
        and anomalies.array.left[0] == seg_len
        and anomalies.array.right[0] == 2 * seg_len
    )


def test_capa_anomalies_segment_length():
    detector = CAPA.create_test_instance()
    min_segment_length = 5
    detector.set_params(
        segment_penalty=0.0,
        min_segment_length=min_segment_length,
    )

    n = 100
    df = generate_alternating_data(n_segments=1, segment_length=n, random_state=13)
    anomalies = detector.fit_predict(df)["ilocs"]

    anomaly_lengths = anomalies.array.right - anomalies.array.left
    assert np.all(anomaly_lengths == 5)


def test_capa_point_anomalies():
    detector = CAPA.create_test_instance()
    n_segments = 2
    seg_len = 50
    p = 3
    df = generate_alternating_data(
        n_segments=n_segments,
        mean=20,
        segment_length=seg_len,
        p=p,
        random_state=134,
    )
    point_anomaly_iloc = 20
    df.iloc[point_anomaly_iloc] += 50

    anomalies = detector.fit_predict(df)
    estimated_point_anomaly_iloc = anomalies["ilocs"].iloc[0]

    assert point_anomaly_iloc == estimated_point_anomaly_iloc.left


def test_capa_errors():
    """Test CAPA error cases."""
    cost = MultivariateGaussianCost([0.0, np.eye(2)])

    # Test point saving must have a minimum size of 1
    with pytest.raises(ValueError):
        CAPA(point_saving=cost)

    # Test min_segment_length must be greater than 2
    with pytest.raises(ValueError):
        CAPA(min_segment_length=1)

    # Test max_segment_length must be greater than min_segment_length
    with pytest.raises(ValueError):
        CAPA(min_segment_length=5, max_segment_length=4)


def test_capa_different_data_shapes():
    """Test CAPA with segment and point savings having different data shapes."""

    # Create detector
    detector = CAPA()
    detector.fit(pd.DataFrame(np.random.randn(10, 2)))

    # Create two PenalisedScore objects with different data shapes
    segment_saving = to_saving(L2Cost(param=0.0))
    point_saving = to_saving(L2Cost(param=0.0))

    segment_data = pd.DataFrame(np.random.randn(20, 2))
    point_data = pd.DataFrame(np.random.randn(30, 2))  # Different number of samples

    segment_penalised_saving = PenalisedScore(segment_saving)
    point_penalised_saving = PenalisedScore(point_saving)

    segment_penalised_saving.fit(segment_data)
    point_penalised_saving.fit(point_data)

    # Test that run_capa raises ValueError due to different shapes
    with pytest.raises(ValueError, match="same number of samples"):
        run_capa(
            segment_penalised_saving=segment_penalised_saving,
            point_penalised_saving=point_penalised_saving,
            min_segment_length=2,
            max_segment_length=10,
        )


def test_invalid_savings():
    """
    Test that CAPA raises an error when given an invalid saving argument.
    """
    with pytest.raises(ValueError, match="segment_saving"):
        CAPA("l2")
    with pytest.raises(ValueError, match="segment_saving"):
        CAPA(ChangeScore(COSTS[3].create_test_instance()))

    score = L2Saving()
    # Simulate a penalised score not constructed by PenalisedScore
    score.set_tags(is_penalised=True)
    with pytest.raises(ValueError, match="penalised"):
        CAPA(segment_saving=score)
    with pytest.raises(ValueError, match="penalised"):
        CAPA(point_saving=score)


def test_valid_point_savings():
    score = L2Saving()
    penalised_score = PenalisedScore(score)
    CAPA(point_saving=penalised_score)
