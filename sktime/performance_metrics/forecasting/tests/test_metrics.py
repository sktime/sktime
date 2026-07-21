# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for some metrics."""
# currently this consists entirely of doctests from _classes and _functions
# since the numpy output print changes between versions

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

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
    y_true_df = pd.DataFrame(y_true, columns=["a", "b"])
    y_pred_df = pd.DataFrame(y_pred, columns=["a", "b"])
    assert np.allclose(
        linex_error.evaluate_by_index(y_true_df, y_pred_df),
        np.array(
            [
                [0.14872127, 0.36787944],
                [0.0, 0.36787944],
                [0.36787944, 0.36787944],
            ]
        ),
    )
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
def test_owa_aggregate_then_ratio():
    """Regression test for OWA using aggregate-then-ratio (M4 definition).

    Naive2 has zero error at one horizon step but non-zero error over the full
    horizon in this test. Averaging per-step metric ratios can produce
    astronomically large values.
    OWA must aggregate MASE and sMAPE over the horizon before dividing as
    per the definition in the M4 competition paper.
    """
    from sktime.performance_metrics.forecasting import OverallWeightedAverage

    if not _check_estimator_deps(OverallWeightedAverage, severity="none"):
        pytest.skip("OverallWeightedAverage dependencies not available.")

    y_train = np.array([100.0, 100.1, 100.0, 100.2, 100.1, 100.0])
    y_true = np.array([100.05, 100.0, 100.15])
    y_pred = np.array([100.0, 100.05, 100.0])

    # Naive2 forecasts for this y_train (SeasonalityACF sp=1, NaiveForecaster last).
    # To check the definition of Naive2, check class OverallWeightedAverage.
    y_pred_naive2 = np.array([100.0, 100.0, 100.1])

    # MASE denominator: mean absolute lag-1 difference in training (sp=1).
    mase_scale = np.abs(np.diff(y_train)).mean()

    model_abs_errors = np.abs(y_true - y_pred)
    naive2_abs_errors = np.abs(y_true - y_pred_naive2)

    mase_model = model_abs_errors.mean() / mase_scale
    mase_naive2 = naive2_abs_errors.mean() / mase_scale
    mase_ratio = mase_model / mase_naive2

    smape_model = np.mean(2 * model_abs_errors / (np.abs(y_true) + np.abs(y_pred)))
    smape_naive2 = np.mean(
        2 * naive2_abs_errors / (np.abs(y_true) + np.abs(y_pred_naive2))
    )
    smape_ratio = smape_model / smape_naive2

    expected_owa = 0.5 * (mase_ratio + smape_ratio)

    owa = OverallWeightedAverage(sp=1)(y_true, y_pred, y_train=y_train)

    assert np.isfinite(owa)
    assert owa < 100
    assert np.allclose(owa, expected_owa)
