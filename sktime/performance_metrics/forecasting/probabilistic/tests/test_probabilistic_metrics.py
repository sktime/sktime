# -*- coding: utf-8 -*-
"""Tests for probabilistic quantiles."""
import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
from sktime.utils._testing.series import _make_series

list_of_metrics = [PinballLoss]

y_uni = _make_series(n_columns=1)
y_train_uni, y_test_uni = temporal_train_test_split(y_uni)
fh_uni = np.arange(len(y_test_uni)) + 1
f_uni = ColumnEnsembleForecaster(ThetaForecaster(sp=12))
f_uni.fit(y_train_uni)

QUANTILE_PRED_UNI = f_uni.predict_quantiles(fh=fh_uni, alpha=[0.5])
INTERVAL_PRED_UNI = f_uni.predict_interval(fh=fh_uni, coverage=0.9)


@pytest.mark.parametrize("score_average", [True, False])
@pytest.mark.parametrize("Metric", list_of_metrics)
@pytest.mark.parametrize("Predictions", [QUANTILE_PRED_UNI, INTERVAL_PRED_UNI])
def test_output(Metric, score_average, Predictions):
    """Test output is correct class."""
    y_true = y_test_uni
    loss = Metric.create_test_instance()
    loss.set_params(score_average=score_average)
    eval_loss = loss.evaluate(y_true, y_pred=Predictions)
    index_loss = loss.evaluate_by_index(y_true, y_pred=Predictions)

    if score_average:
        assert isinstance(eval_loss, float)
        assert isinstance(index_loss, pd.Series)
    else:
        assert isinstance(eval_loss, pd.Series)
        assert isinstance(index_loss, pd.DataFrame)


# QUANTILE ONLY
@pytest.mark.parametrize("Metric", list_of_metrics)
@pytest.mark.parametrize("Predictions", [QUANTILE_PRED_UNI])
def test_evaluate_to_zero(Metric, Predictions):
    """Tests whether metric returns 0 when y_true=y_pred."""
    loss = Metric.create_test_instance()
    y_true = Predictions
    eval_loss = loss.evaluate(y_true, y_pred=Predictions)
    assert np.isclose(0, eval_loss).all()


@pytest.mark.parametrize("Metric", list_of_metrics)
@pytest.mark.parametrize("Predictions", [QUANTILE_PRED_UNI])
def test_evaluate_alpha_postive(Metric, Predictions):
    """Tests whether metric returns 0 when y_true=y_pred by index."""
    Loss = Metric.create_test_instance().set_params(alpha=0.5)
    res = Loss(y_true=y_test_uni, y_pred=Predictions)
    res


@pytest.mark.parametrize("Metric", list_of_metrics)
@pytest.mark.parametrize("Predictions", [QUANTILE_PRED_UNI])
def test_evaluate_alpha_negative(Metric, Predictions):
    """Tests whether metric returns 0 when y_true=y_pred by index."""
    with pytest.raises(ValueError):
        Loss = Metric.create_test_instance().set_params(alpha=0.05)
        res = Loss(y_true=y_test_uni, y_pred=Predictions)
        res
