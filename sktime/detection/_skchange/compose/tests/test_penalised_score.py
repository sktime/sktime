"""Tests for penalised scores."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection._skchange.change_scores import CUSUM, MultivariateGaussianScore
from sktime.detection._skchange.compose.penalised_score import PenalisedScore


def test_penalised_score_init():
    scorer = CUSUM()
    penalised_score = PenalisedScore(scorer)
    assert penalised_score._get_required_cut_size() == scorer._get_required_cut_size()

    with pytest.raises(ValueError, match="penalised"):
        PenalisedScore(PenalisedScore(scorer))

    with pytest.raises(ValueError):
        scorer = MultivariateGaussianScore()
        PenalisedScore(scorer, np.array([1.0, 2.0]))


def test_penalised_score_fit():
    scorer = CUSUM()

    df3 = pd.DataFrame(np.random.randn(100, 3))

    # Runs with a constant penalty
    penalised_score = PenalisedScore(scorer, 10)
    penalised_score.fit(df3, scorer)
    assert penalised_score.is_fitted

    # Raises error when penalty does not match the number of columns
    penalised_score = PenalisedScore(scorer, np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        penalised_score.fit(df3)


def test_penalised_score_get_model_size():
    scorer = CUSUM()
    penalised_score = PenalisedScore(scorer)
    assert penalised_score.get_model_size(1) == scorer.get_model_size(1)
    assert penalised_score.get_model_size(5) == scorer.get_model_size(5)
