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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_mse_evaluate_by_index_rmse_pseudo_values():
    """Regression: RMSE evaluate_by_index returns jackknife pseudo-values.

    Tests that RMSE evaluate_by_index returns jackknife pseudo-values,
    not raw squared errors.

    This test defines the intended behaviour after the refactor of
    MeanSquaredError(square_root=True)._evaluate_by_index, to guard against
    future regressions.
    """
    from sktime.performance_metrics.forecasting import MeanSquaredError

    # simple 1D example
    y_true = pd.Series([3.0, -0.5, 2.0, 7.0], name="y")
    y_pred = pd.Series([2.5, 0.0, 2.0, 8.0], name="yhat")

    mse = MeanSquaredError(square_root=True)

    # value under test
    by_index = mse.evaluate_by_index(y_true, y_pred)
    # cast to numpy 1D
    by_index = np.asarray(by_index)

    # independently compute expected jackknife pseudo-values for RMSE
    # this mirrors the logic in _mse.py
    raw_values = (y_true - y_pred) ** 2
    raw_values = raw_values.to_numpy()
    n = raw_values.shape[0]

    mse_all = raw_values.mean()
    rmse_all = np.sqrt(mse_all)
    sqe_sum = raw_values.sum()
    mse_jackknife = (sqe_sum - raw_values) / (n - 1)
    rmse_jackknife = np.sqrt(mse_jackknife)
    expected = n * rmse_all - (n - 1) * rmse_jackknife

    np.testing.assert_allclose(by_index, expected)
