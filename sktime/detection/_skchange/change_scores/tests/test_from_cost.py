import numpy as np
import pytest

from sktime.detection._skchange.anomaly_scores import L2Saving
from sktime.detection._skchange.change_scores import ChangeScore, to_change_score
from sktime.detection._skchange.costs import COSTS
from sktime.detection._skchange.costs.base import BaseCost
from sktime.detection._skchange.tests.test_all_interval_scorers import (
    skip_if_no_test_data,
)


@pytest.mark.parametrize("cost_class", COSTS)
def test_change_score_with_costs(cost_class: type[BaseCost]):
    cost_instance = cost_class.create_test_instance()
    change_score = ChangeScore(cost=cost_instance)
    skip_if_no_test_data(change_score)
    X = np.random.randn(100, 1)
    change_score.fit(X)
    cuts = np.array([[0, 50, 100]])
    scores = change_score._evaluate(cuts)
    assert scores.shape == (1, 1)


@pytest.mark.parametrize("evaluator", COSTS)
def test_to_change_score(evaluator: type[BaseCost]):
    cost_instance = evaluator.create_test_instance()
    change_score = to_change_score(cost_instance)
    assert isinstance(change_score, ChangeScore)


def test_to_change_score_invalid():
    with pytest.raises(ValueError):
        to_change_score("invalid_evaluator")
    with pytest.raises(ValueError):
        to_change_score(L2Saving())
