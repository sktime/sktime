# -*- coding: utf-8 -*-
"""Tests for probabilistic quantiles."""
import numpy as np
import pandas as pd
import pytest

from sktime.utils._testing.series import _make_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.performance_metrics.forecasting.probabilistic import PinballLoss, EmpiricalCoverage, ConstraintViolation

list_of_metrics = [PinballLoss, EmpiricalCoverage]

# test data
#y = np.log1p(load_airline())
y = _make_series(n_columns=1)
y_train, y_test = temporal_train_test_split(y)
fh = np.arange(len(y_test)) + 1
f = ColumnEnsembleForecaster(ThetaForecaster(sp=12))
f.fit(y_train)

"""
score average = TRUE/FALSE
multivariable = TRUE/FALSE
multiscores = TRUE/FALSE

Data types
Univariate and single score
Univariate and multi score

Multivariate and single score
Multivariate and multiscore

For each of the data types we need to test with score average = T/F \
    and multioutput with "raw_values" and "uniform_average"
"""
QUANTILE_PRED_UNI_S = f.predict_quantiles(fh=fh, alpha=[0.5])
INTERVAL_PRED_UNI_S = f.predict_interval(fh=fh, coverage=0.9)

QUANTILE_PRED_UNI_M = f.predict_quantiles(fh-fh, alpha=[0.05, 0.5, 0.95])
INTERVAL_PRED_UNI_M = f.predict_interval(fh=fh, coverage=[0.7, 0.8, 0.9, 0.99])


@pytest.mark.parametrize("y_pred", [QUANTILE_PRED_UNI_S, INTERVAL_PRED_UNI_S])
@pytest.mark.parametrize("metric", list_of_metrics)
def test_output_univariate(metric, y_pred):
    """Test output is correct format."""
    y_true = y_test
    loss = metric.create_test_instance()
    eval_loss = loss.evaluate(y_true, y_pred=y_pred)
    index_loss = loss.evaluate_by_index(y_true, y_pred=y_pred)

    assert isinstance(eval_loss, float)
    assert isinstance(index_loss, pd.DataFrame)

    # same number of rows, one col since univariate
    assert index_loss.shape == (y_true.shape[0], 1)
    assert index_loss.index == y_true.index



@pytest.mark.parametrize("y_pred", [QUANTILE_PRED_UNI_S, INTERVAL_PRED_UNI_S])
@pytest.mark.parametrize("metric", list_of_metrics)
def test_output_no_avg(metric, y_pred):
    """Test output is correct format."""
    y_true = y_test
    loss = metric.create_test_instance()
    loss.set_param(score_average=False)
    eval_loss = loss.evaluate(y_true, y_pred=y_pred)
    index_loss = loss.evaluate_by_index(y_true, y_pred=y_pred)

    if score_average:
        assert isinstance(eval_loss, float)
        assert isinstance(index_loss, pd.Series)
    else:
        assert isinstance(eval_loss, pd.DataFrame)
        assert isinstance(index_loss, pd.DataFrame)

    expected_index = y_pred.index.get_level_values([0 ,1]).unique()
    ncol = len(expected_index)

    # same number of rows, one col per variable and quantile/coverage value
    assert eval_loss.shape == (1, ncol)
    assert eval_loss.shape[0] == 0
    assert eval_loss.index == expected_index

    # same number of rows, one col per variable and quantile/coverage value
    assert index_loss.shape == (y_true.shape[0], ncol)
    assert index_loss.index == y_true.index
    assert index_loss.index == expected_index


@pytest.mark.parametrize("y_pred", [QUANTILE_PRED_UNI, INTERVAL_PRED_UNI])
@pytest.mark.parametrize("Metric", list_of_metrics)
def test_output_multivariate_default(metric, y_pred):
    """Test output is correct format."""
    y_true = y_test
    loss = metric.create_test_instance()
    eval_loss = loss.evaluate(y_true, y_pred=y_pred)
    index_loss = loss.evaluate_by_index(y_true, y_pred=y_pred)

    assert isinstance(eval_loss, float)
    assert isinstance(index_loss, pd.DataFrame)

    # same number of rows, one col since univariate
    assert index_loss.shape == (y_true.shape[0], 1)
    assert index_loss.index == y_true.index


@pytest.mark.parametrize("y_pred", [QUANTILE_PRED_UNI, INTERVAL_PRED_UNI])
@pytest.mark.parametrize("Metric", list_of_metrics)
def test_output_multivariate_no_avg(metric, y_pred):
    """Test output is correct format."""
    y_true = y_test
    loss = metric.create_test_instance()
    eval_loss = loss.evaluate(y_true, y_pred=y_pred)
    index_loss = loss.evaluate_by_index(y_true, y_pred=y_pred)

    assert isinstance(eval_loss, float)
    assert isinstance(index_loss, pd.DataFrame)

    # same number of rows, one col since univariate
    assert index_loss.shape == (y_true.shape[0], 1)
    assert index_loss.index == y_true.index





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
    index_loss = Loss.evaluate_by_index(y_true, y_pred=QUANTILE_PRED)
    assert all([np.isclose(0, a) for a in index_loss.values[:, 0]])


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
