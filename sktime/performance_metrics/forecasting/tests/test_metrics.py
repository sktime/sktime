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
def test_msmape_class():
    """Test MeanAbsolutePercentageErrorStabilized (msMAPE).

    Hand-computed expected values for basic, single-observation,
    all-zeros, and multivariate cases.
    """
    from sktime.performance_metrics.forecasting import (
        MeanAbsolutePercentageErrorStabilized,
    )

    # --- basic case ---
    y_true = np.array([3, 5, 2, 7])
    y_pred = np.array([2.5, 4, 2, 8])
    metric = MeanAbsolutePercentageErrorStabilized()
    assert np.allclose(metric(y_true, y_pred), 0.13004235907461714)

    # --- single observation (S_1 = 0, reduces to sMAPE-like term) ---
    y_true_single = np.array([5.0])
    y_pred_single = np.array([3.0])
    assert np.allclose(metric(y_true_single, y_pred_single), 0.5)

    # --- all zeros: numerator is zero so result is zero ---
    y_true_zeros = np.array([0.0, 0.0, 0.0])
    y_pred_zeros = np.array([0.0, 0.0, 0.0])
    assert np.allclose(metric(y_true_zeros, y_pred_zeros), 0.0)

    # --- multivariate with raw_values ---
    y_true_mv = pd.DataFrame({"A": [3, 5, 2], "B": [10, 20, 30]})
    y_pred_mv = pd.DataFrame({"A": [2.5, 4, 2], "B": [11, 19, 31]})
    metric_raw = MeanAbsolutePercentageErrorStabilized(multioutput="raw_values")
    result = metric_raw(y_true_mv, y_pred_mv)
    expected = np.array([0.13468013468013468, 0.058229720201551184])
    assert np.allclose(result, expected)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_nmse_class():
    """Test NormalizedMeanSquaredError (NMSE).

    Hand-computed expected values for basic, constant-series, and multivariate cases.
    """
    from sktime.performance_metrics.forecasting import NormalizedMeanSquaredError

    # --- basic case ---
    y_true = np.array([3, -0.5, 2, 7, 2])
    y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    metric = NormalizedMeanSquaredError()
    assert np.allclose(metric(y_true, y_pred), 0.2630806138733395)

    # --- perfect forecast ---
    y_pred_perfect = y_true.copy()
    assert np.allclose(metric(y_true, y_pred_perfect), 0.0)

    # --- constant series: var=0 -> result is large but finite ---
    y_const = np.array([5.0, 5.0, 5.0])
    y_pred_const = np.array([4.0, 6.0, 5.5])
    result = metric(y_const, y_pred_const)
    assert np.isfinite(result)
    assert result > 0

    # --- multivariate with raw_values ---
    y_true_mv = pd.DataFrame({"A": [3, -0.5, 2, 7, 2], "B": [1, 2, 3, 4, 5]})
    y_pred_mv = pd.DataFrame(
        {"A": [2.5, 0.0, 2, 8, 1.25], "B": [1.5, 2.5, 2.5, 3.5, 5.5]}
    )
    metric_raw = NormalizedMeanSquaredError(multioutput="raw_values")
    result = metric_raw(y_true_mv, y_pred_mv)
    expected = np.array([0.2630806138733395, 0.3535533905932738])
    assert np.allclose(result, expected)

    # --- uniform_average multivariate ---
    metric_avg = NormalizedMeanSquaredError()
    result = metric_avg(y_true_mv, y_pred_mv)
    assert np.allclose(result, 0.3083170022333066)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_iqre_class():
    """Test InterQuartileRangeError (IQR).

    Hand-computed expected values for basic, constant-series, and multivariate cases.
    """
    from sktime.performance_metrics.forecasting import InterQuartileRangeError

    # --- basic case ---
    y_true = np.array([3, -0.5, 2, 7, 2])
    y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    metric = InterQuartileRangeError()
    assert np.allclose(metric(y_true, y_pred), 0.6422616289332564)

    # --- perfect forecast ---
    y_pred_perfect = y_true.copy()
    assert np.allclose(metric(y_true, y_pred_perfect), 0.0)

    # --- constant series: IQR=0 -> result is large but finite ---
    y_const = np.array([5.0, 5.0, 5.0])
    y_pred_const = np.array([4.0, 6.0, 5.5])
    result = metric(y_const, y_pred_const)
    assert np.isfinite(result)
    assert result > 0

    # --- multivariate with raw_values ---
    y_true_mv = pd.DataFrame({"A": [3, -0.5, 2, 7, 2], "B": [1, 2, 3, 4, 5]})
    y_pred_mv = pd.DataFrame(
        {"A": [2.5, 0.0, 2, 8, 1.25], "B": [1.5, 2.5, 2.5, 3.5, 5.5]}
    )
    metric_raw = InterQuartileRangeError(multioutput="raw_values")
    result = metric_raw(y_true_mv, y_pred_mv)
    expected = np.array([0.6422616289332564, 0.25])
    assert np.allclose(result, expected)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_kln_class():
    """Test KLDivergenceNormal (KL-N).

    Hand-computed expected values for basic and perfect-forecast cases.
    Uses test data where first two predictions exactly match to avoid
    numerical blow-up from eps-clamped rolling variance at early indices.
    """
    from sktime.performance_metrics.forecasting import KLDivergenceNormal

    # --- basic case ---
    y_true = np.array([3.0, 5.0, 2.0, 7.0, 4.0, 6.0])
    y_pred = np.array([3.0, 5.0, 3.0, 6.0, 5.0, 5.5])
    metric = KLDivergenceNormal()
    assert np.allclose(metric(y_true, y_pred), 0.5771341616115743)

    # --- perfect forecast ---
    y_pred_perfect = y_true.copy()
    assert np.allclose(metric(y_true, y_pred_perfect), 0.0)

    # --- with custom eps ---
    metric_eps = KLDivergenceNormal(eps=1e-6)
    result = metric_eps(y_true, y_pred)
    assert np.isfinite(result)
    assert result > 0

    # --- with rolling window (window=3) ---
    metric_w3 = KLDivergenceNormal(window=3)
    result_w3 = metric_w3(y_true, y_pred)
    assert np.isfinite(result_w3)
    assert result_w3 > 0
    # rolling window should differ from expanding window on this data
    assert not np.allclose(result_w3, metric(y_true, y_pred))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_klde2_class():
    """Test KLDivergenceDoubleExponential (KL-DE2).

    Hand-computed expected values for basic and perfect-forecast cases.
    Uses test data where first two predictions exactly match to avoid
    numerical blow-up from eps-clamped rolling MAD at early indices.
    """
    from sktime.performance_metrics.forecasting import KLDivergenceDoubleExponential

    # --- basic case ---
    y_true = np.array([3.0, 5.0, 2.0, 7.0, 4.0, 6.0])
    y_pred = np.array([3.0, 5.0, 3.0, 6.0, 5.0, 5.5])
    metric = KLDivergenceDoubleExponential()
    assert np.allclose(metric(y_true, y_pred), 0.14407771573805175)

    # --- perfect forecast ---
    y_pred_perfect = y_true.copy()
    assert np.allclose(metric(y_true, y_pred_perfect), 0.0)

    # --- with custom eps ---
    metric_eps = KLDivergenceDoubleExponential(eps=1e-6)
    result = metric_eps(y_true, y_pred)
    assert np.isfinite(result)
    assert result > 0

    # --- with rolling window (window=3) ---
    metric_w3 = KLDivergenceDoubleExponential(window=3)
    result_w3 = metric_w3(y_true, y_pred)
    assert np.isfinite(result_w3)
    assert result_w3 > 0
    # rolling window should differ from expanding window on this data
    assert not np.allclose(result_w3, metric(y_true, y_pred))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_klde1_class():
    """Test KLDivergenceSingleExponential (KL-DE1).

    Hand-computed expected values for basic and perfect-forecast cases.
    Uses test data where first two predictions exactly match to avoid
    numerical blow-up from eps-clamped rolling std at early indices.
    """
    from sktime.performance_metrics.forecasting import (
        KLDivergenceDoubleExponential,
        KLDivergenceSingleExponential,
    )

    # --- basic case ---
    y_true = np.array([3.0, 5.0, 2.0, 7.0, 4.0, 6.0])
    y_pred = np.array([3.0, 5.0, 3.0, 6.0, 5.0, 5.5])
    metric = KLDivergenceSingleExponential()
    assert np.allclose(metric(y_true, y_pred), 0.1285730069501481)

    # --- perfect forecast ---
    y_pred_perfect = y_true.copy()
    assert np.allclose(metric(y_true, y_pred_perfect), 0.0)

    # --- with custom eps ---
    metric_eps = KLDivergenceSingleExponential(eps=1e-6)
    result = metric_eps(y_true, y_pred)
    assert np.isfinite(result)
    assert result > 0

    # --- differs from KL-DE2 (different scale estimator) ---
    klde2 = KLDivergenceDoubleExponential()
    assert not np.allclose(metric(y_true, y_pred), klde2(y_true, y_pred))

    # --- with rolling window (window=3) ---
    metric_w3 = KLDivergenceSingleExponential(window=3)
    result_w3 = metric_w3(y_true, y_pred)
    assert np.isfinite(result_w3)
    assert result_w3 > 0
    # rolling window should differ from expanding window on this data
    assert not np.allclose(result_w3, metric(y_true, y_pred))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_theilu2_class():
    """Test TheilU2 (Theil's U2 statistic).

    Hand-computed expected values for basic, perfect-forecast, and
    naive-equivalent cases.
    """
    from sktime.performance_metrics.forecasting import TheilU2

    # --- basic case ---
    y_train = np.array([5.0, 0.5, 4.0, 6.0, 3.0, 5.0, 2.0])
    y_true = np.array([3.0, -0.5, 2.0, 7.0, 2.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0, 1.25])
    metric = TheilU2()
    result = metric(y_true, y_pred, y_train=y_train)
    assert np.isfinite(result)
    assert result > 0
    assert np.allclose(result, 0.17226798597767884)

    # --- perfect forecast -> 0.0 ---
    y_pred_perfect = y_true.copy()
    assert np.allclose(metric(y_true, y_pred_perfect, y_train=y_train), 0.0)

    # --- forecast equal to naive -> 1.0 ---
    # naive forecast with sp=1: previous actual value
    y_naive_pred = np.concatenate([y_train[-1:], y_true[:-1]])
    result_naive = metric(y_true, y_naive_pred, y_train=y_train)
    assert np.allclose(result_naive, 1.0)

    # --- sp=2 variant ---
    metric_sp2 = TheilU2(sp=2)
    result_sp2 = metric_sp2(y_true, y_pred, y_train=y_train)
    assert np.isfinite(result_sp2)
    assert result_sp2 > 0


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_evaluate_by_index_returns_correct_index():
    """Test that evaluate_by_index returns results with the correct index.

    Metrics that use jackknife pseudo-values (NMSE, IQRE, KL-N, TheilU2) must
    return a pd.Series/DataFrame whose index matches y_true's index.
    """
    from sktime.performance_metrics.forecasting import (
        InterQuartileRangeError,
        KLDivergenceNormal,
        NormalizedMeanSquaredError,
        TheilU2,
    )

    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    y_true = pd.DataFrame({"a": [3.0, 5.0, 2.0, 7.0, 4.0, 6.0]}, index=idx)
    y_pred = pd.DataFrame({"a": [3.0, 5.0, 3.0, 6.0, 5.0, 5.5]}, index=idx)

    for MetricCls in [NormalizedMeanSquaredError, InterQuartileRangeError,
                      KLDivergenceNormal]:
        metric = MetricCls()
        result = metric.evaluate_by_index(y_true, y_pred)
        assert result.index.equals(idx), (
            f"{MetricCls.__name__}.evaluate_by_index returned wrong index"
        )
        assert len(result) == len(y_true)

    # TheilU2 requires y_train
    y_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=range(5))
    metric_u2 = TheilU2()
    result_u2 = metric_u2.evaluate_by_index(y_true, y_pred, y_train=y_train)
    assert result_u2.index.equals(idx), (
        "TheilU2.evaluate_by_index returned wrong index"
    )
    assert len(result_u2) == len(y_true)
