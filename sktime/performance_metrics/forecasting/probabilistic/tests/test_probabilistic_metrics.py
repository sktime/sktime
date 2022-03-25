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


@pytest.mark.parametrize("Metric", list_of_metrics)
def test_output(Metric):
    """Test output is correct class."""
    y_true = y_test
    Loss = Metric.create_test_instance()
    eval_loss = Loss.evaluate(y_true, y_pred=QUANTILE_PRED)
    index_loss = Loss.evaluate_by_index(y_true, y_pred=QUANTILE_PRED)

    assert isinstance(eval_loss, pd.DataFrame)
    assert isinstance(index_loss, pd.DataFrame)


@pytest.mark.parametrize("Metric", list_of_metrics)
def test_evaluate_to_zero(Metric):
    """Tests whether metric returns 0 when y_true=y_pred."""
    Loss = Metric.create_test_instance()
    y_true = QUANTILE_PRED["Quantiles"][0.5]
    eval_loss = Loss.evaluate(y_true, y_pred=QUANTILE_PRED)
    assert np.isclose(0, eval_loss).all()


@pytest.mark.parametrize("Metric", list_of_metrics)
def test_evaluate_by_index_to_zero(Metric):
    """Tests whether metric returns 0 when y_true=y_pred by index."""
    Loss = Metric.create_test_instance()
    y_true = QUANTILE_PRED["Quantiles"][0.5]
    index_loss = Loss.evaluate_by_index(y_true, y_pred=QUANTILE_PRED)
    assert all([np.isclose(0, a) for a in index_loss.values[:, 0]])
