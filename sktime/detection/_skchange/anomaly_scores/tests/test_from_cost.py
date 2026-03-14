import numpy as np
import pytest

from sktime.detection._skchange.anomaly_scores import (
    LocalAnomalyScore,
    Saving,
    to_local_anomaly_score,
    to_saving,
)
from sktime.detection._skchange.change_scores import CUSUM
from sktime.detection._skchange.costs import COSTS, L2Cost
from sktime.detection._skchange.costs.base import BaseCost
from sktime.detection._skchange.costs.tests.test_all_costs import create_fixed_cost_test_instance
from sktime.detection._skchange.tests.test_all_interval_scorers import skip_if_no_test_data


@pytest.mark.parametrize("CostClass", COSTS)
def test_saving_init(CostClass: type[BaseCost]):
    if not CostClass.get_class_tag("supports_fixed_param"):
        pytest.skip(f"{CostClass.__name__} does not support fixed parameters.")
    baseline_cost = create_fixed_cost_test_instance(CostClass)

    saving = Saving(baseline_cost)
    assert saving.baseline_cost == baseline_cost
    assert saving.optimised_cost.param is None


@pytest.mark.parametrize("CostClass", COSTS)
def test_saving_min_size(CostClass: type[BaseCost]):
    if not CostClass.get_class_tag("supports_fixed_param"):
        pytest.skip(f"{CostClass.__name__} does not support fixed parameters.")

    cost = create_fixed_cost_test_instance(CostClass)
    saving = Saving(baseline_cost=cost)

    assert saving.min_size == cost.min_size

    skip_if_no_test_data(saving)
    np.random.seed(132)
    X = np.random.randn(100, 1)
    cost.fit(X)
    saving.fit(X)
    assert saving.min_size == cost.min_size


@pytest.mark.parametrize("CostClass", COSTS)
def test_saving_fit(CostClass: type[BaseCost]):
    if not CostClass.get_class_tag("supports_fixed_param"):
        pytest.skip(f"{CostClass.__name__} does not support fixed parameters.")

    cost = create_fixed_cost_test_instance(CostClass)
    saving = Saving(baseline_cost=cost)
    skip_if_no_test_data(saving)

    X = np.random.randn(100, 1)
    saving.fit(X)
    assert saving.baseline_cost_.is_fitted
    assert saving.optimised_cost_.is_fitted


@pytest.mark.parametrize("CostClass", COSTS)
def test_saving_evaluate(CostClass: type[BaseCost]):
    if not CostClass.get_class_tag("supports_fixed_param"):
        pytest.skip(f"{CostClass.__name__} does not support fixed parameters.")

    cost = create_fixed_cost_test_instance(CostClass)
    saving = Saving(baseline_cost=cost)
    skip_if_no_test_data(saving)
    X = np.random.randn(100, 1)
    saving.fit(X)
    intervals = np.array([[0, 15], [10, 25], [25, 40]])
    savings = saving.evaluate(intervals)
    assert savings.shape == (3, 1)


@pytest.mark.parametrize("CostClass", COSTS)
def test_to_saving_raises_with_no_param_set(CostClass: type[BaseCost]):
    """Test that to_saving raises ValueError with BaseCost that has no param set."""
    if not CostClass.get_class_tag("supports_fixed_param"):
        pytest.skip(f"{CostClass.__name__} does not support fixed parameters.")

    with pytest.raises(ValueError, match="fixed param"):
        cost = CostClass.create_test_instance()
        cost = cost.set_params(param=None)
        to_saving(cost)


def test_to_saving_raises_without_fixed_params_support():
    """Test that `to_saving` raises ValueError when `supports_fixed_param` is False."""
    with pytest.raises(
        ValueError, match="The baseline cost must support fixed parameter"
    ):
        # BaseCost does not support fixed parameters, by default.
        cost = BaseCost()
        to_saving(cost)


def test_to_saving_error():
    with pytest.raises(ValueError):
        to_saving("invalid_evaluator")
    with pytest.raises(ValueError):
        to_saving(CUSUM())


@pytest.mark.parametrize("CostClass", COSTS)
def test_to_local_anomaly_score_with_base_cost(CostClass: type[BaseCost]):
    if not CostClass.get_class_tag("supports_fixed_param"):
        pytest.skip(f"{CostClass.__name__} does not support fixed parameters.")

    cost = create_fixed_cost_test_instance(CostClass)
    local_anomaly_score = to_local_anomaly_score(cost)
    assert isinstance(local_anomaly_score, LocalAnomalyScore)
    assert local_anomaly_score.cost == cost


@pytest.mark.parametrize("CostClass", COSTS)
def test_to_local_anomaly_score_with_local_anomaly_score(CostClass: type[BaseCost]):
    if not CostClass.get_class_tag("supports_fixed_param"):
        pytest.skip(f"{CostClass.__name__} does not support fixed parameters.")

    cost = create_fixed_cost_test_instance(CostClass)
    local_anomaly_score_instance = LocalAnomalyScore(cost=cost)
    result = to_local_anomaly_score(local_anomaly_score_instance)
    assert result is local_anomaly_score_instance


@pytest.mark.parametrize("CostClass", COSTS)
def test_local_anomaly_score_evaluate(CostClass: type[BaseCost]):
    if not CostClass.get_class_tag("supports_fixed_param"):
        pytest.skip(f"{CostClass.__name__} does not support fixed parameters.")

    cost = create_fixed_cost_test_instance(CostClass)
    local_anomaly_score = LocalAnomalyScore(cost=cost)
    skip_if_no_test_data(local_anomaly_score)

    X = np.random.randn(100, 4)  # Need to be 4 columns for LinearRegressionCost.
    local_anomaly_score.fit(X)
    cuts = np.array([[0, 15, 30, 45], [5, 20, 35, 50], [10, 25, 40, 55]])
    scores = local_anomaly_score.evaluate(cuts)
    assert scores.shape[0] == 3


def test_to_local_anomaly_score_error():
    with pytest.raises(ValueError):
        to_local_anomaly_score("invalid_evaluator")
    with pytest.raises(ValueError):
        to_local_anomaly_score(CUSUM())


class HighMinSizeL2Cost(L2Cost):
    """A custom cost class that changes `min_size` to return a tuple."""

    @property
    def min_size(self) -> int:
        """Return a tuple instead of an integer."""
        return 3


def test_raises_if_inner_if_inner_interval_size_is_too_small():
    """Test LocalAnomalyScore raises ValueError if inner interval size is too small."""
    local_score = to_local_anomaly_score(scorer=HighMinSizeL2Cost(param=0.0))
    local_score.fit(np.random.randn(10, 4))  # Fit to have _X set.
    with pytest.raises(
        ValueError, match="The inner intervals must be at least min_size="
    ):
        local_score._check_cuts(
            np.array([[1, 3, 5, 8]])
        )  # Too small inner interval size.


def test_raises_if_inner_if_surrounding_interval_size_is_too_small():
    """Test LocalAnomalyScore raises ValueError if inner interval size is too small."""
    local_score = to_local_anomaly_score(scorer=HighMinSizeL2Cost(param=0.0))
    local_score.fit(np.random.randn(10, 4))  # Fit to have _X set.
    with pytest.raises(
        ValueError, match="The surrounding intervals must be at least min_size="
    ):
        local_score._check_cuts(
            np.array([[1, 2, 6, 7]])
        )  # Too small surrounding interval size.
