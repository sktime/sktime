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


# TODO 1.5.0: remove this test
@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics", "sktime.base"]),
    reason="run test only if performance_metrics or base module has changed",
)
@pytest.mark.parametrize(
    "old_tag,new_tag,true_cls_name,false_cls_name",
    [
        (
            "requires-y-train",
            "requires_y_train",
            "MeanAbsoluteScaledError",
            "MeanSquaredError",
        ),
        (
            "requires-y-pred-benchmark",
            "requires_y_pred_benchmark",
            "RelativeLoss",
            "MeanSquaredError",
        ),
    ],
)
def test_deprecated_metric_tags(old_tag, new_tag, true_cls_name, false_cls_name):
    """Test deprecation and aliasing of dashed metric tags."""
    import warnings

    import sktime.performance_metrics.forecasting as fc_metrics

    true_cls = getattr(fc_metrics, true_cls_name)
    false_cls = getattr(fc_metrics, false_cls_name)
    true_est = true_cls()
    false_est = false_cls()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", FutureWarning)
        assert true_cls.get_class_tag(new_tag) is True
        assert false_cls.get_class_tag(new_tag) is False
        assert true_est.get_tag(new_tag) is True
        assert false_est.get_tag(new_tag) is False
        false_est.set_tags(**{new_tag: True})
        assert false_est.get_tag(new_tag) is True
    assert not [w for w in record if issubclass(w.category, FutureWarning)]

    msg = rf"{old_tag}.*1\.5\.0.*{new_tag}"
    with pytest.warns(FutureWarning, match=msg):
        assert true_cls.get_class_tag(old_tag) is True
    with pytest.warns(FutureWarning, match=msg):
        assert true_est.get_tag(old_tag) is True
    with pytest.warns(FutureWarning, match=msg):
        override = false_cls().set_tags(**{old_tag: True})
    assert override.get_tag(new_tag) is True

    class _LegacyMetric(false_cls):
        _tags = {old_tag: True}

    assert _LegacyMetric.get_class_tag(new_tag) is True
    with pytest.warns(FutureWarning, match=old_tag):
        assert _LegacyMetric.get_class_tag(old_tag) is True
