import numpy as np
import pytest
from skbase._exceptions import NotFittedError

from sktime.detection._skchange.costs import COSTS, RankCost
from sktime.detection._skchange.costs.base import BaseCost
from sktime.detection._skchange.datasets import generate_alternating_data
from sktime.detection._skchange.tests.test_all_interval_scorers import skip_if_no_test_data


def find_fixed_param_combination(
    cost_class: type[BaseCost] | BaseCost,
) -> dict[str, float]:
    """Find the first fixed parameter combination in the test parameters of a cost."""
    test_param_sets = cost_class.get_test_params()
    fixed_test_param_set = None
    for param_set in test_param_sets:
        if "param" in param_set and param_set["param"] is not None:
            fixed_test_param_set = param_set
            break

    if fixed_test_param_set is None:
        class_name = (
            cost_class.__name__
            if isinstance(cost_class, type)
            else cost_class.__class__.__name__
        )
        raise ValueError(
            f"No fixed `param` argument found in `get_test_params()` of"
            f" the cost class {class_name}"
        )

    return fixed_test_param_set


def create_fixed_cost_test_instance(cost_class: type[BaseCost]) -> BaseCost:
    """Create a fixed instance of the cost class."""
    fixed_param = find_fixed_param_combination(cost_class)
    return cost_class.create_test_instance().set_params(**fixed_param)


def test_find_fixed_param_combination_value_error():
    class MockCost:
        @staticmethod
        def get_test_params():
            return [{"param": None}, {"param": None}]

    with pytest.raises(ValueError):
        find_fixed_param_combination(MockCost)


@pytest.mark.parametrize("CostClass", COSTS)
def test_l2_cost_init(CostClass: type[BaseCost]):
    cost = CostClass.create_test_instance()
    assert cost.param is None


@pytest.mark.parametrize("CostClass", COSTS)
def test_expected_cut_entries(CostClass: type[BaseCost]):
    cost = CostClass.create_test_instance()
    assert cost._get_required_cut_size() == 2


@pytest.mark.parametrize("CostClass", COSTS)
def test_cost_evaluation_optim_gt_fixed(CostClass: type[BaseCost]):
    if not CostClass.get_class_tag("supports_fixed_param"):
        pytest.skip(f"{CostClass.__name__} does not support fixed parameters.")

    optim_cost = CostClass.create_test_instance()
    skip_if_no_test_data(optim_cost)
    fixed_cost = create_fixed_cost_test_instance(CostClass)
    np.random.seed(1001)
    X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=40)
    optim_cost.fit(X)
    fixed_cost.fit(X)
    intervals = np.array([[0, 15], [10, 30], [5, 25], [25, 40]])
    optim_costs = optim_cost.evaluate(intervals)
    fixed_costs = fixed_cost.evaluate(intervals)
    assert np.all(optim_costs <= fixed_costs)


@pytest.mark.parametrize("CostClass", COSTS)
def test_cost_evaluation_positive(CostClass: type[BaseCost]):
    cost = CostClass.create_test_instance()
    skip_if_no_test_data(cost)
    if isinstance(cost, RankCost):
        pytest.skip("RankCost produces negative costs.")

    n = 50
    df = generate_alternating_data(n_segments=1, segment_length=n, p=1, random_state=5)
    cost.fit(df)
    starts = np.arange(n - 15)
    ends = np.repeat(n - 1, len(starts))
    intervals = np.column_stack((starts, ends))
    costs = cost.evaluate(intervals)
    assert np.all(costs >= 0.0)


@pytest.mark.parametrize("CostClass", COSTS)
def test_accessing_n_samples_before_fit_raises(
    CostClass: type[BaseCost],
):
    """Test that accessing n_samples before fitting raises an error."""
    cost = CostClass.create_test_instance()
    with pytest.raises(NotFittedError):
        cost.n_samples


@pytest.mark.parametrize("CostClass", COSTS)
def test_accessing_n_variables_before_fit_raises(
    CostClass: type[BaseCost],
):
    """Test that accessing n_variables before fitting raises an error."""
    cost = CostClass.create_test_instance()
    with pytest.raises(NotFittedError):
        cost.n_variables
