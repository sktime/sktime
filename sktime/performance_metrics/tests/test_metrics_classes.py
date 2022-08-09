# -*- coding: utf-8 -*-
"""Tests for classes in _classes module."""
from inspect import getmembers, isclass

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.performance_metrics.forecasting import (
    MeanSquaredError,
    _classes,
    make_forecasting_scorer,
)
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series

metric_classes = getmembers(_classes, isclass)

exclude_starts_with = ("_", "Base", "Vectorized")
metric_classes = [x for x in metric_classes if not x[0].startswith(exclude_starts_with)]

names, metrics = zip(*metric_classes)


@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
@pytest.mark.parametrize("metric", metrics, ids=names)
def test_metric_output_direct(metric, multioutput, n_columns):
    """Test output is of correct type, dependent on multioutput.

    Also tests that four ways to call the metric yield equivalent results:
        1. using the __call__ dunder
        2. calling the evaluate method
    """
    y_pred = _make_series(n_columns=n_columns, n_timepoints=20, random_state=21)
    y_true = _make_series(n_columns=n_columns, n_timepoints=20, random_state=42)

    # coerce to DataFrame since _make_series does not return consisten output type
    y_pred = pd.DataFrame(y_pred)
    y_true = pd.DataFrame(y_true)

    res = dict()

    res[1] = metric(multioutput=multioutput)(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_benchmark=y_pred,
        y_train=y_true,
    )

    res[2] = metric(multioutput=multioutput).evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_benchmark=y_pred,
        y_train=y_true,
    )

    if multioutput == "uniform_average":
        assert all(isinstance(x, float) for x in res.values())
    elif multioutput == "raw_values":
        assert all(isinstance(x, np.ndarray) for x in res.values())
        assert all(x.ndim == 1 for x in res.values())
        assert all(len(x) == len(y_true.columns) for x in res.values())

    # assert results from all options are equal
    assert np.allclose(res[1], res[2])


@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize(
    "multilevel", ["uniform_average", "uniform_average_time", "raw_values"]
)
@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
def test_metric_hierarchical(multioutput, multilevel, n_columns):
    """Test hierarchical input for metrics."""
    y_pred = _make_hierarchical(random_state=21, n_columns=n_columns)
    y_true = _make_hierarchical(random_state=42, n_columns=n_columns)

    metric = MeanSquaredError(multioutput=multioutput, multilevel=multilevel)

    res = metric(
        y_true=y_true,
        y_pred=y_pred,
    )

    if multilevel == "raw_values":
        assert isinstance(res, (pd.DataFrame, pd.Series))
        assert isinstance(res.index, pd.MultiIndex)

        expected_index = y_true.index.droplevel(-1).unique()
        found_index = res.index.unique()
        assert set(expected_index) == set(found_index)
        if multioutput == "raw_values" and isinstance(res, pd.DataFrame):
            assert all(y_true.columns == res.columns)
    # if multilevel == "uniform_average" or "uniform_average_time"
    else:
        if multioutput == "uniform_average":
            assert isinstance(res, float)
        elif multioutput == "raw_values":
            assert isinstance(res, np.ndarray)
            assert res.ndim == 1
            assert len(res) == len(y_true.columns)


@pytest.mark.parametrize("greater_is_better", [True, False])
def test_custom_metric(greater_is_better):
    """Test custom metric constructor, integration _DynamicForecastingErrorMetric."""
    from sktime.utils.estimator_checks import check_estimator

    y = load_airline()

    def custom_mape(y_true, y_pred) -> float:

        eps = np.finfo(np.float64).eps

        result = np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps))

        return float(result)

    fc_scorer = make_forecasting_scorer(
        func=custom_mape,
        name="custom_mape",
        greater_is_better=False,
    )

    assert isinstance(fc_scorer, _classes._DynamicForecastingErrorMetric)

    score = fc_scorer(y, y)
    assert isinstance(score, float)

    check_estimator(fc_scorer, return_exceptions=False)
