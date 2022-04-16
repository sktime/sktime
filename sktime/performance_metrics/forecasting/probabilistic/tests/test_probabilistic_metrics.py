# -*- coding: utf-8 -*-
"""Tests for probabilistic quantiles."""
import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting.probabilistic import PinballLoss

list_of_metrics = [PinballLoss]

# test data
y = np.log1p(load_airline())
y_train, y_test = temporal_train_test_split(y)
fh = np.arange(len(y_test)) + 1
f = ThetaForecaster(sp=12)
f.fit(y_train)


QUANTILE_PRED = f.predict_quantiles(fh=fh, alpha=[0.5])
INTERVAL_PRED = f.predict_interval(fh=fh, coverage=0.9)


@pytest.mark.parametrize("score_average", [True, False])
@pytest.mark.parametrize("Metric", list_of_metrics)
def test_output(Metric, score_average):
    """Test output is correct class."""
    y_true = y_test
    loss = Metric.create_test_instance()
    loss.set_params(score_average=score_average)
    eval_loss = loss.evaluate(y_true, y_pred=QUANTILE_PRED)
    index_loss = loss.evaluate_by_index(y_true, y_pred=QUANTILE_PRED)

    if score_average:
        assert isinstance(eval_loss, float)
        assert isinstance(index_loss, pd.Series)
    else:
        assert isinstance(eval_loss, pd.DataFrame)
        assert isinstance(index_loss, pd.DataFrame)


@pytest.mark.parametrize("Metric", list_of_metrics)
def test_evaluate_to_zero(Metric):
    """Tests whether metric returns 0 when y_true=y_pred."""
    loss = Metric.create_test_instance()
    y_true = QUANTILE_PRED["Quantiles"][0.5]
    eval_loss = loss.evaluate(y_true, y_pred=QUANTILE_PRED)
    assert np.isclose(0, eval_loss).all()


@pytest.mark.parametrize("Metric", list_of_metrics)
def test_evaluate_by_index_to_zero(Metric):
    """Tests whether metric returns 0 when y_true=y_pred by index."""
    loss = Metric.create_test_instance()
    y_true = QUANTILE_PRED["Quantiles"][0.5]
    index_loss = loss.evaluate_by_index(y_true, y_pred=QUANTILE_PRED)
    assert all(np.isclose(0, a) for a in index_loss)


@pytest.mark.parametrize("Metric", list_of_metrics)
def test_evaluate_alpha_postive(Metric):
    """Tests whether metric returns 0 when y_true=y_pred by index."""
    Loss = Metric.create_test_instance().set_params(alpha=0.5)
    res = Loss(y_true=y_test, y_pred=QUANTILE_PRED)
    res


@pytest.mark.parametrize("Metric", list_of_metrics)
def test_evaluate_alpha_negative(Metric):
    """Tests whether metric returns 0 when y_true=y_pred by index."""
    with pytest.raises(ValueError):
        Loss = Metric.create_test_instance().set_params(alpha=0.05)
        res = Loss(y_true=y_test, y_pred=QUANTILE_PRED)
        res
