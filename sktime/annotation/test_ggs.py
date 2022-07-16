# coding: utf-8
from .ggs import GGS, GGSEstimator
import numpy as np
import pytest


@pytest.fixture
def univariate_mean_shift():
    x = np.concatenate(tuple(np.ones(5) * i ** 2 for i in range(4)))
    return x[:, np.newaxis]


def test_GGS_find_change_points(univariate_mean_shift):

    ggs = GGS(k_max=10, lamb=1.0)
    pred = ggs.find_change_points(univariate_mean_shift)
    assert isinstance(pred, list)
    assert len(pred) == 5


def test_GGSEstimator(univariate_mean_shift):

    ggs = GGSEstimator(k_max=5, lamb=0.5)
    assert ggs.get_params() == {
        "k_max": 5,
        "lamb": 0.5,
        "verbose": False,
        "max_shuffles": 250,
        "random_state": None,
    }
