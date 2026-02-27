# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for some metrics."""
# currently this consists entirely of doctests from _classes and _functions
# since the numpy output print changes between versions

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_gmse_class():
    """Doctest from GeometricMeanSquaredError."""
    from sktime.performance_metrics.forecasting import GeometricMeanSquaredError

    y_true = np.array([3, -0.5, 2, 7, 2])
    y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    gmse = GeometricMeanSquaredError()

    assert np.allclose(gmse(y_true, y_pred), 2.80399089461488e-07)
    rgmse = GeometricMeanSquaredError(square_root=True)
    assert np.allclose(rgmse(y_true, y_pred), 0.000529527232030127)

    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    gmse = GeometricMeanSquaredError()
    assert np.allclose(gmse(y_true, y_pred), 0.5000000000115499)
    rgmse = GeometricMeanSquaredError(square_root=True)
    assert np.allclose(rgmse(y_true, y_pred), 0.5000024031086919)
    gmse = GeometricMeanSquaredError(multioutput="raw_values")
    assert np.allclose(gmse(y_true, y_pred), np.array([2.30997255e-11, 1.00000000e00]))
    rgmse = GeometricMeanSquaredError(multioutput="raw_values", square_root=True)
    assert np.allclose(rgmse(y_true, y_pred), np.array([4.80621738e-06, 1.00000000e00]))
    gmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7])
    assert np.allclose(gmse(y_true, y_pred), 0.7000000000069299)
    rgmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7], square_root=True)
    assert np.allclose(rgmse(y_true, y_pred), 0.7000014418652152)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_gmse_function():
    """Doctest from geometric_mean_squared_error."""
    from sktime.performance_metrics.forecasting import geometric_mean_squared_error

    gmse = geometric_mean_squared_error
    y_true = np.array([3, -0.5, 2, 7, 2])
    y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    assert np.allclose(gmse(y_true, y_pred), 2.80399089461488e-07)
    assert np.allclose(gmse(y_true, y_pred, square_root=True), 0.000529527232030127)
    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    assert np.allclose(gmse(y_true, y_pred), 0.5000000000115499)
    assert np.allclose(gmse(y_true, y_pred, square_root=True), 0.5000024031086919)
    assert np.allclose(
        gmse(y_true, y_pred, multioutput="raw_values"),
        np.array([2.30997255e-11, 1.00000000e00]),
    )
    assert np.allclose(
        gmse(y_true, y_pred, multioutput="raw_values", square_root=True),
        np.array([4.80621738e-06, 1.00000000e00]),
    )
    assert np.allclose(gmse(y_true, y_pred, multioutput=[0.3, 0.7]), 0.7000000000069299)
    assert np.allclose(
        gmse(y_true, y_pred, multioutput=[0.3, 0.7], square_root=True),
        0.7000014418652152,
    )

    assert np.allclose(
        gmse(
            np.array([1, 2, 3]), np.array([6, 5, 4]), horizon_weight=np.array([7, 8, 9])
        ),
        6.185891035775025,
    )


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_linex_class():
    """Doctest from MeanLinexError."""
    from sktime.performance_metrics.forecasting import MeanLinexError

    linex_error = MeanLinexError()
    y_true = np.array([3, -0.5, 2, 7, 2])
    y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    assert np.allclose(linex_error(y_true, y_pred), 0.19802627763937575)
    linex_error = MeanLinexError(b=2)
    assert np.allclose(linex_error(y_true, y_pred), 0.3960525552787515)
    linex_error = MeanLinexError(a=-1)
    assert np.allclose(linex_error(y_true, y_pred), 0.2391800623225643)
    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    linex_error = MeanLinexError()
    assert np.allclose(linex_error(y_true, y_pred), 0.2700398392309829)
    linex_error = MeanLinexError(a=-1)
    assert np.allclose(linex_error(y_true, y_pred), 0.49660966225813563)
    linex_error = MeanLinexError(multioutput="raw_values")
    assert np.allclose(linex_error(y_true, y_pred), np.array([0.17220024, 0.36787944]))
    linex_error = MeanLinexError(multioutput=[0.3, 0.7])
    assert np.allclose(linex_error(y_true, y_pred), 0.30917568000716666)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_linex_function():
    """Doctest from mean_linex_error."""
    from sktime.performance_metrics.forecasting import mean_linex_error

    y_true = np.array([3, -0.5, 2, 7, 2])
    y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    assert np.allclose(mean_linex_error(y_true, y_pred), 0.19802627763937575)
    assert np.allclose(mean_linex_error(y_true, y_pred, b=2), 0.3960525552787515)
    assert np.allclose(mean_linex_error(y_true, y_pred, a=-1), 0.2391800623225643)
    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    assert np.allclose(mean_linex_error(y_true, y_pred), 0.2700398392309829)
    assert np.allclose(mean_linex_error(y_true, y_pred, a=-1), 0.49660966225813563)
    assert np.allclose(
        mean_linex_error(y_true, y_pred, multioutput="raw_values"),
        np.array([0.17220024, 0.36787944]),
    )
    assert np.allclose(
        mean_linex_error(y_true, y_pred, multioutput=[0.3, 0.7]), 0.30917568000716666
    )


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_make_scorer():
    """Test make_forecasting_scorer and the failure case in #4827."""
    import functools

    from sklearn.metrics import mean_tweedie_deviance

    from sktime.performance_metrics.forecasting import make_forecasting_scorer

    rmsle = functools.partial(mean_tweedie_deviance, power=1.5)

    scorer = make_forecasting_scorer(rmsle, name="MTD")

    scorer.evaluate(pd.Series([1, 2, 3]), pd.Series([1, 2, 4]))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_make_scorer_sklearn():
    """Test make_forecasting_scorer and the failure case in #5715.

    Naive adaptation fails on newer sklearn versions due to
    decoration with sklearn's custom input constraint wrapper.
    """
    from sklearn.metrics import mean_absolute_error

    from sktime.performance_metrics.forecasting import make_forecasting_scorer

    scorer = make_forecasting_scorer(mean_absolute_error, name="MAE")

    scorer.evaluate(pd.Series([1, 2, 3]), pd.Series([1, 2, 4]))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_metric_coercion_bug():
    """Tests for sensible output when using hierarchical arg with non-hierarchical data.

    Failure case in bug #6413.
    """
    from sktime.performance_metrics.forecasting import MeanAbsoluteError

    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    mae = MeanAbsoluteError(multilevel="raw_values", multioutput=[0.4, 0.6])
    metric = mae(y_true, y_pred)

    assert isinstance(metric, pd.DataFrame)
    assert metric.shape == (1, 1)


# ---------------------------------------------------------------------------
# Regression tests for gh-5102
# MASE / RMSSE must work when y_train has NaN due to unequal-length series
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize(
    "metric_fn, kwargs",
    [
        ("mean_squared_scaled_error", {"square_root": True}),
        ("mean_squared_scaled_error", {"square_root": False}),
        ("mean_absolute_scaled_error", {}),
        ("median_absolute_scaled_error", {}),
        ("median_squared_scaled_error", {"square_root": True}),
        ("median_squared_scaled_error", {"square_root": False}),
    ],
)
def test_scaled_metrics_unequal_length_no_error(metric_fn, kwargs):
    """Regression test for gh-5102.

    Scaled metrics must not raise when y_train contains NaN values due to
    unequal-length series padded into a rectangular (wide-format) array.
    """
    import importlib

    mod = importlib.import_module("sktime.performance_metrics.forecasting")
    fn = getattr(mod, metric_fn)

    # Series 0 has 4 training points (no NaN)
    # Series 1 has 3 training points (padded with 1 leading NaN)
    y_train = np.array([[1.5, np.nan], [0.5, 1.0], [-1.0, 1.0], [7.0, -6.0]])
    y_true = np.array([[0.5, 1.0], [-1.0, 1.0], [7.0, -6.0]])
    y_pred = np.array([[0.0, 2.0], [-1.0, 2.0], [8.0, -5.0]])

    # Must not raise — this was the bug reported in gh-5102
    result = fn(y_true, y_pred, y_train=y_train, **kwargs)

    assert np.isfinite(result), (
        f"{metric_fn} returned non-finite result for unequal-length y_train: {result}"
    )


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize(
    "metric_fn, kwargs",
    [
        ("mean_squared_scaled_error", {"square_root": True}),
        ("mean_absolute_scaled_error", {}),
    ],
)
def test_scaled_metrics_unequal_length_multioutput_raw(metric_fn, kwargs):
    """gh-5102: multioutput='raw_values' must return finite per-column array."""
    import importlib

    mod = importlib.import_module("sktime.performance_metrics.forecasting")
    fn = getattr(mod, metric_fn)

    y_train = np.array([[1.5, np.nan], [0.5, 1.0], [-1.0, 1.0], [7.0, -6.0]])
    y_true = np.array([[0.5, 1.0], [-1.0, 1.0], [7.0, -6.0]])
    y_pred = np.array([[0.0, 2.0], [-1.0, 2.0], [8.0, -5.0]])

    result = fn(
        y_true, y_pred, y_train=y_train, multioutput="raw_values", **kwargs
    )

    assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
    assert np.all(np.isfinite(result)), (
        f"{metric_fn}: expected all-finite result with raw_values, got {result}"
    )


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_scaled_metrics_nan_free_unchanged():
    """gh-5102: fix must not change output for NaN-free (equal-length) inputs."""
    from sktime.performance_metrics.forecasting import (
        MeanAbsoluteScaledError,
        MeanSquaredScaledError,
    )

    # All four columns have the same length — no NaN
    y_train = np.array([[1.5, 0.5], [0.5, 1.0], [-1.0, 1.0], [7.0, -6.0]])
    y_true = np.array([[0.5, 1.0], [-1.0, 1.0], [7.0, -6.0]])
    y_pred = np.array([[0.0, 2.0], [-1.0, 2.0], [8.0, -5.0]])

    result_mase = MeanAbsoluteScaledError()(y_true, y_pred, y_train=y_train)
    result_rmsse = MeanSquaredScaledError(square_root=True)(
        y_true, y_pred, y_train=y_train
    )

    assert np.isfinite(result_mase), f"MASE not finite for clean data: {result_mase}"
    assert np.isfinite(result_rmsse), (
        f"RMSSE not finite for clean data: {result_rmsse}"
    )


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_scaled_metrics_univariate_still_works():
    """gh-5102: fix must not break the standard univariate (1-D) path."""
    from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

    y_train = np.array([5.0, 0.5, 4.0, 6.0, 3.0, 5.0, 2.0])
    y_true = np.array([3.0, -0.5, 2.0, 7.0, 2.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0, 1.25])

    result = MeanAbsoluteScaledError()(y_true, y_pred, y_train=y_train)

    assert np.isfinite(result), f"MASE not finite for univariate input: {result}"
    # Value should match the docstring example exactly
    assert np.isclose(result, 0.18333333333333335, rtol=1e-6), (
        f"Unexpected MASE value for univariate: {result}"
    )
