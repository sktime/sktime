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
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.panel import _make_panel
from sktime.utils._testing.series import _make_series
from sktime.utils.parallel import _get_parallel_test_fixtures

metric_classes = getmembers(_classes, isclass)

exclude_starts_with = ("_", "Base", "Vectorized")
metric_classes = [x for x in metric_classes if not x[0].startswith(exclude_starts_with)]

names, metrics = zip(*metric_classes)

MULTIOUTPUT = ["uniform_average", "raw_values", "numpy"]

# list of parallelization backends to test
BACKENDS = _get_parallel_test_fixtures("config")


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("multioutput", MULTIOUTPUT)
@pytest.mark.parametrize("metric", metrics, ids=names)
def test_metric_output_direct(metric, multioutput, n_columns):
    """Test output is of correct type, dependent on multioutput.

    Also tests that four ways to call the metric yield equivalent results:
        1. using the __call__ dunder
        2. calling the evaluate method
    """
    # create numpy weights based on n_columns
    if multioutput == "numpy":
        if n_columns == 1:
            return None
        multioutput = np.random.rand(n_columns)

    # create test data
    y_pred = _make_series(n_columns=n_columns, n_timepoints=20, random_state=21)
    y_true = _make_series(n_columns=n_columns, n_timepoints=20, random_state=42)

    # coerce to DataFrame since _make_series does not return consistent output type
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

    if isinstance(multioutput, np.ndarray) or multioutput == "uniform_average":
        assert all(isinstance(x, float) for x in res.values())
    elif multioutput == "raw_values":
        assert all(isinstance(x, np.ndarray) for x in res.values())
        assert all(x.ndim == 1 for x in res.values())
        assert all(len(x) == len(y_true.columns) for x in res.values())

    # assert results from all options are equal
    assert np.allclose(res[1], res[2])


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

    check_estimator(fc_scorer, raise_exceptions=True)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("multioutput", MULTIOUTPUT)
@pytest.mark.parametrize("metric", metrics, ids=names)
def test_metric_output_by_instance(metric, multioutput, n_columns):
    """Test output of evaluate_by_index is of correct type, dependent on multioutput."""
    # create numpy weights based on n_columns
    if multioutput == "numpy":
        if n_columns == 1:
            return None
        multioutput = np.random.rand(n_columns)

    # create test data
    y_pred = _make_series(n_columns=n_columns, n_timepoints=20, random_state=21)
    y_true = _make_series(n_columns=n_columns, n_timepoints=20, random_state=42)

    # coerce to DataFrame since _make_series does not return consistent output type
    y_pred = pd.DataFrame(y_pred)
    y_true = pd.DataFrame(y_true)

    res = metric(multioutput=multioutput).evaluate_by_index(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_benchmark=y_pred,
        y_train=y_true,
    )

    if isinstance(multioutput, str) and multioutput == "raw_values":
        assert isinstance(res, pd.DataFrame)
        assert (res.columns == y_true.columns).all()
    else:
        assert isinstance(res, pd.Series)

    assert (res.index == y_true.index).all()


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("multilevel", ["uniform_average", "raw_values"])
@pytest.mark.parametrize("multioutput", MULTIOUTPUT)
def test_metric_hierarchical_by_index(multioutput, multilevel, n_columns, backend):
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

    res = metric.evaluate_by_index(
        y_true=y_true,
        y_pred=y_pred,
    )

    if isinstance(multioutput, str) and multioutput == "raw_values":
        assert isinstance(res, pd.DataFrame)
        assert (res.columns == y_true.columns).all()
    else:
        assert isinstance(res, pd.Series)

    if multilevel == "raw_values":
        assert isinstance(res.index, pd.MultiIndex)
        expected_index = y_true.index
    else:
        assert not isinstance(res.index, pd.MultiIndex)
        expected_index = y_true.index.get_level_values(-1).unique()

    found_index = res.index.unique()
    assert set(expected_index) == set(found_index)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize("metric", metrics, ids=names)
def test_uniform_average_time(metric):
    """Tests that uniform_average_time indeed ignores index."""
    y_true = _make_panel()
    y_pred = _make_panel()

    metric_obj = metric(multilevel="uniform_average_time")

    y_true_noix = y_true.reset_index(drop=True)
    y_pred_noix = y_pred.reset_index(drop=True)

    res = metric_obj.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_benchmark=y_pred,
        y_train=y_true,
    )

    res_noix = metric_obj.evaluate(
        y_true=y_true_noix,
        y_pred=y_pred_noix,
        y_pred_benchmark=y_pred_noix,
        y_train=y_true_noix,
    )

    assert np.allclose(res, res_noix)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize("metric", metrics, ids=names)
def test_metric_weights(metric):
    """Test that weights are correctly applied to the metric."""
    y_true = np.array([3, -0.5, 2, 7, 2])
    y_pred = np.array([2.5, 0.5, 2, 8, 2.25])
    wts = np.array([0.1, 0.2, 0.1, 0.3, 2.4])

    y_kwargs = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_benchmark": y_true,
        "y_train": y_true,
    }

    metric_obj = metric()
    if metric_obj(**y_kwargs) == metric_obj(sample_weight=wts, **y_kwargs):
        raise ValueError(f"Metric {metric} does not handle sample_weight correctly")

    # wt_metr = metric(sample_weight=wts)
    # res_wt = wt_metr(y_true, y_pred)
    # assert np.allclose(res_wt, metric_obj(y_true, y_pred, sample_weight=wts))
