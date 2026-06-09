import numpy as np
import pytest

from sktime.detection._skchange.base import BaseIntervalScorer
from sktime.detection._skchange.change_scores import CHANGE_SCORES, to_change_score
from sktime.detection._skchange.compose.penalised_score import PenalisedScore
from sktime.detection._skchange.costs import COSTS
from sktime.detection._skchange.costs.tests.test_all_costs import (
    create_fixed_cost_test_instance,
)
from sktime.detection._skchange.datasets import generate_alternating_data
from sktime.detection._skchange.tests.test_all_interval_scorers import (
    skip_if_no_test_data,
)

COST_INSTANCES = [
    to_change_score(create_fixed_cost_test_instance(cost))
    for cost in COSTS
    if cost.create_test_instance().get_tag("supports_fixed_param")
]
SCORE_INSTANCES = [score.create_test_instance() for score in CHANGE_SCORES]
PENALISED_SCORE_INSTANCES = [
    PenalisedScore(score)
    for score in COST_INSTANCES + SCORE_INSTANCES
    if not score.get_tag("is_penalised")
]
ALL_SCORE_INSTANCES = COST_INSTANCES + SCORE_INSTANCES + PENALISED_SCORE_INSTANCES


@pytest.mark.parametrize("change_score", ALL_SCORE_INSTANCES)
def test_scores(change_score: BaseIntervalScorer):
    """Test all available changepoint scores."""
    skip_if_no_test_data(change_score)

    n = 50
    df = generate_alternating_data(n_segments=1, segment_length=n, p=2, random_state=5)
    change_score.fit(df)
    splits = np.arange(15, n - 15, dtype=int).reshape(-1, 1)
    cuts = np.column_stack(
        (np.zeros(splits.shape, dtype=int), splits, np.full(splits.shape, n, dtype=int))
    )
    scores = change_score.evaluate(cuts)

    score_is_penalised = change_score.get_tag("is_penalised")
    min_value = 0.0 if not score_is_penalised else -np.inf
    assert np.all(scores >= min_value)
