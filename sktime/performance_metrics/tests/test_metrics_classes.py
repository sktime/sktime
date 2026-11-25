"""Tests for classes in forecasting module."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.performance_metrics.forecasting import (
    MeanSquaredError,
    make_forecasting_scorer,
)
from sktime.performance_metrics.forecasting._base import _DynamicForecastingErrorMetric
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.parallel import _get_parallel_test_fixtures

MULTIOUTPUT = ["uniform_average", "raw_values", "numpy"]

# list of parallelization backends to test
BACKENDS = _get_parallel_test_fixtures("config")


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize(
    "multilevel", ["uniform_average", "uniform_average_time", "raw_values"]
)
@pytest.mark.parametrize("multioutput", MULTIOUTPUT)
def test_metric_hierarchical(multioutput, multilevel, n_columns, backend):
    """Test hierarchical input for metrics."""
    # create numpy weights based on n_columns
    if multioutput == "numpy":
        if n_columns == 1:
            return None
        multioutput = np.random.rand(n_columns)

    # create test data
    y_pred = _make_hierarchical(random_state=21, n_columns=n_columns)
    y_true = _make_hierarchical(random_state=42, n_columns=n_columns)

    metric = MeanSquaredError(multioutput=multioutput, multilevel=multilevel)
    metric.set_config(**backend)

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
        if isinstance(multioutput, str):
            if multioutput == "raw_values" and isinstance(res, pd.DataFrame):
                assert all(y_true.columns == res.columns)
    # if multilevel == "uniform_average" or "uniform_average_time"
    else:
        if isinstance(multioutput, np.ndarray) or multioutput == "uniform_average":
            assert isinstance(res, float)
        elif multioutput == "raw_values":
            assert isinstance(res, np.ndarray)
            assert res.ndim == 1
            assert len(res) == len(y_true.columns)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize("greater_is_better", [True, False])
def test_custom_metric(greater_is_better):
    """Test custom metric constructor, integration _DynamicForecastingErrorMetric."""
    from sktime.utils.estimator_checks import check_estimator

    y = load_airline()

    def custom_mape(y_true, y_pred) -> float:
        eps = np.finfo(np.float64).eps

        mapes = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)
        result = np.mean(mapes, axis=0)

        return result

    fc_scorer = make_forecasting_scorer(
        func=custom_mape,
        name="custom_mape",
        greater_is_better=greater_is_better,
    )

    assert isinstance(fc_scorer, _DynamicForecastingErrorMetric)

    score = fc_scorer(y, y)
    assert isinstance(score, float)

    check_estimator(fc_scorer, raise_exceptions=True)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_metric_reordered_multiindex():
    """Test forecasting metrics when y_true and y_pred have different index order.

    Fixes issue #7779.
    """
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.performance_metrics.forecasting import MeanAbsoluteError

    # Series 1 ends at 4 -> forecast is 4
    # Series 2 ends at 8 -> forecast is 8
    past_dataset = pd.DataFrame(
        {
            "series_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "time_id": [0, 1, 2, 3, 0, 1, 2, 3],
            "y": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    ).set_index(["series_id", "time_id"])

    # Future dataset has series_id ordered [2, 1] with asymmetric errors:
    # Series 2 future values are 18 -> true error is |18 - 8| = 10.0
    # Series 1 future values are 5  -> true error is |5 - 4| = 1.0
    future_dataset = pd.DataFrame(
        {"series_id": [2, 2, 1, 1], "time_id": [4, 5, 4, 5], "y": [18, 18, 5, 5]}
    ).set_index(["series_id", "time_id"])

    model = NaiveForecaster()
    model.fit(y=past_dataset)
    forecasts = model.predict(fh=[1, 2])

    evaluation_metric = MeanAbsoluteError()
    with pytest.warns(UserWarning, match="do not have the same row index order"):
        metric_score = evaluation_metric(future_dataset, forecasts)

    # Average error: (1.0 + 1.0 + 10.0 + 10.0) / 4 = 5.5
    # (If misaligned, error would be (|18-4|*2 + |5-8|*2)/4 = 34/4 = 8.5)
    assert np.isclose(metric_score, 5.5)

    # Verify per-series level evaluation via multilevel='raw_values'
    raw_metric = MeanAbsoluteError(multilevel="raw_values")
    raw_scores = raw_metric(future_dataset, forecasts)
    assert np.isclose(raw_scores.loc[1].iloc[0], 1.0)
    assert np.isclose(raw_scores.loc[2].iloc[0], 10.0)

    # Verify evaluate_by_index preserves instance-level alignment
    by_index = raw_metric.evaluate_by_index(future_dataset, forecasts)
    assert np.allclose(by_index.loc[1].values, 1.0)
    assert np.allclose(by_index.loc[2].values, 10.0)

    # Verify default evaluate_by_index averages over series at each timepoint
    by_index_avg = evaluation_metric.evaluate_by_index(future_dataset, forecasts)
    assert np.allclose(by_index_avg.values, 5.5)
