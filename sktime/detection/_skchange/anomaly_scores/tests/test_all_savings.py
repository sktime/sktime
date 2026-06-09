import numpy as np
import pytest

from sktime.detection._skchange.anomaly_scores import SAVINGS, to_saving
from sktime.detection._skchange.compose.penalised_score import PenalisedScore
from sktime.detection._skchange.costs import COSTS
from sktime.detection._skchange.costs.tests.test_all_costs import create_fixed_cost_test_instance
from sktime.detection._skchange.datasets import generate_alternating_data
from sktime.detection._skchange.tests.test_all_interval_scorers import skip_if_no_test_data

COST_INSTANCES = [
    to_saving(create_fixed_cost_test_instance(cost))
    for cost in COSTS
    if cost.create_test_instance().get_tag("supports_fixed_param")
]
SAVING_INSTANCES = [saving.create_test_instance() for saving in SAVINGS]
PENALISED_SAVING_INSTANCES = [
    PenalisedScore(saving)
    for saving in COST_INSTANCES + SAVING_INSTANCES
    if not saving.get_tag("is_penalised")
]
ALL_SAVING_INSTANCES = COST_INSTANCES + SAVING_INSTANCES + PENALISED_SAVING_INSTANCES


@pytest.mark.parametrize("saving", ALL_SAVING_INSTANCES)
def test_saving_values(saving):
    """Test all available savings."""
    skip_if_no_test_data(saving)

    n = 50
    df = generate_alternating_data(n_segments=1, segment_length=n, p=1, random_state=5)
    saving.fit(df)

    starts = np.arange(n - 15)
    ends = np.repeat(n - 1, len(starts))
    intervals = np.column_stack((starts, ends))
    saving_values = saving.evaluate(intervals)

    saving_is_penalised = saving.get_tag("is_penalised")
    min_value = 0.0 if not saving_is_penalised else -np.inf
    assert np.all(saving_values >= min_value)
