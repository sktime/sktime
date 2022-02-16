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


QUANTILE_PRED = f.predict_quantiles(fh=fh, alpha=[0.05, 0.5, 0.95])
INTERVAL_PRED = f.predict_interval(fh=fh, coverage=0.9)
y_true = QUANTILE_PRED["Quantiles"][0.5]

@pytest.mark.parametrize("test_number", [2])
def test_number_is_2(test_number):
    assert test_number == 2

@pytest.mark.parametrize("Metric", list_of_metrics)
def test_output(Metric):
    Loss = Metric.create_test_instance()
    eval_loss = Loss.evaluate(y_true, y_pred=QUANTILE_PRED)
    index_loss = Loss.evaluate_by_index(y_true, y_pred=QUANTILE_PRED)

    assert isinstance(eval_loss, float)
    assert isinstance(index_loss, pd.Series)

@pytest.mark.parametrize("Metric", list_of_metrics)
def test_evaluate_to_zero(Metric):
    """Tests whether metric returns 0 when y_true=y_pred"""
    Loss = Metric.create_test_instance()
    y_true = QUANTILE_PRED["Quantiles"][0.5]
    eval_loss = Loss.evaluate(y_true, y_pred=QUANTILE_PRED)
    assert np.isclose(0, eval_loss)

@pytest.mark.parametrize("Metric", list_of_metrics)
def test_evaluate_by_index_to_zero(Metric):
    """Tests whether metric returns 0 when y_true=y_pred"""
    Loss = Metric.create_test_instance()
    y_true = QUANTILE_PRED["Quantiles"][0.5]
    index_loss = Loss.evaluate_by_index(y_true, y_pred=QUANTILE_PRED)

    assert all([a == 0 for a in index_loss])