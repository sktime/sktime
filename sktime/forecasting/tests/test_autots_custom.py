"""Tests for AutoTS custom functionality."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.autots import AutoTS


@pytest.mark.skipif(
    not _check_estimator_deps(AutoTS, severity="none"),
    reason="autots not available",
)
def test_autots_prediction_intervals():
    """Test that AutoTS can predict intervals."""
    from sktime.datasets import load_airline

    y = load_airline()

    coverage = 0.9
    forecaster = AutoTS(
        model_list="superfast",
        max_generations=1,
        num_validations=0,
        prediction_interval=coverage,
        random_seed=42,
    )

    forecaster.fit(y, fh=[1, 2, 3])

    intervals = forecaster.predict_interval(coverage=coverage)

    assert isinstance(intervals, pd.DataFrame)
    assert intervals.shape == (3, 2)
    assert intervals.columns.nlevels == 3

    lower = intervals.iloc[:, 0]
    upper = intervals.iloc[:, 1]
    assert (upper >= lower).all()

    coverages = [0.9, 0.5]
    intervals_multi = forecaster.predict_interval(coverage=coverages)
    assert intervals_multi.shape == (3, 4)

    expected_cols = pd.MultiIndex.from_product(
        [["Number of airline passengers"], [0.5, 0.9], ["lower", "upper"]],
        names=["variable", "coverage", "lower/upper"],
    )
    intervals_multi = intervals_multi.sort_index(axis=1)
    pd.testing.assert_index_equal(intervals_multi.columns, expected_cols)


@pytest.mark.skipif(
    not _check_estimator_deps(AutoTS, severity="none"),
    reason="autots not available",
)
def test_autots_tags():
    """Test that AutoTS has correct tags."""
    forecaster = AutoTS()
    assert forecaster.get_tag("capability:pred_int") is True
    assert forecaster.get_tag("capability:multivariate") is True
    assert forecaster.get_tag("capability:exogenous") is True


@pytest.mark.skipif(
    not _check_estimator_deps(AutoTS, severity="none"),
    reason="autots not available",
)
def test_autots_exogenous():
    """Test that AutoTS handles exogenous data in fit and predict."""
    from sktime.datasets import load_airline

    y = load_airline()
    X = pd.DataFrame(
        np.random.randint(0, 100, size=(len(y), 2)),
        index=y.index,
        columns=["ex1", "ex2"],
    )

    forecaster = AutoTS(
        model_list="superfast",
        max_generations=1,
        num_validations=0,
        random_seed=42,
    )

    fh = [1, 2, 3]
    forecaster.fit(y, X=X, fh=fh)

    last_idx = y.index[-1]
    if isinstance(last_idx, pd.Period):
        future_idx = pd.period_range(start=last_idx + 1, periods=3, freq=y.index.freq)
    else:
        future_idx = pd.date_range(
            start=last_idx, periods=3 + 1, freq=y.index.freq
        )[1:]

    X_pred = pd.DataFrame(
        np.random.randint(0, 100, size=(3, 2)),
        index=future_idx,
        columns=["ex1", "ex2"],
    )

    y_pred = forecaster.predict(fh=fh, X=X_pred)
    assert len(y_pred) == 3

    intervals = forecaster.predict_interval(fh=fh, X=X_pred, coverage=0.9)
    assert len(intervals) == 3


@pytest.mark.skipif(
    not _check_estimator_deps(AutoTS, severity="none"),
    reason="autots not available",
)
def test_autots_predict_quantiles():
    """Test that AutoTS can produce quantile forecasts."""
    from sktime.datasets import load_airline

    y = load_airline()

    forecaster = AutoTS(
        model_list="superfast",
        max_generations=1,
        num_validations=0,
        random_seed=42,
    )

    forecaster.fit(y, fh=[1, 2, 3])

    alpha = [0.05, 0.5, 0.95]
    quantiles = forecaster.predict_quantiles(alpha=alpha)

    assert isinstance(quantiles, pd.DataFrame)
    assert quantiles.shape == (3, 3)
    assert quantiles.columns.nlevels == 2

    q05 = quantiles.iloc[:, 0]
    q50 = quantiles.iloc[:, 1]
    q95 = quantiles.iloc[:, 2]
    assert (q50 >= q05).all()
    assert (q95 >= q50).all()


@pytest.mark.skipif(
    not _check_estimator_deps(AutoTS, severity="none"),
    reason="autots not available",
)
def test_autots_predict_quantiles_median():
    """Test quantile prediction with alpha=[0.5] only (solo point forecast path)."""
    from sktime.datasets import load_airline

    y = load_airline()

    forecaster = AutoTS(
        model_list="superfast",
        max_generations=1,
        num_validations=0,
        random_seed=42,
    )

    forecaster.fit(y, fh=[1, 2, 3])

    quantiles = forecaster.predict_quantiles(alpha=[0.5])

    assert isinstance(quantiles, pd.DataFrame)
    assert quantiles.shape == (3, 1)
    assert quantiles.columns.nlevels == 2

    y_pred = forecaster.predict()
    np.testing.assert_array_almost_equal(
        quantiles.values.ravel(), y_pred.values.ravel()
    )


@pytest.mark.skipif(
    not _check_estimator_deps(AutoTS, severity="none"),
    reason="autots not available",
)
def test_autots_multivariate_with_exogenous():
    """Test AutoTS with multivariate y and exogenous X."""
    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    y = pd.DataFrame(
        np.random.randn(n, 2),
        index=idx,
        columns=["series_a", "series_b"],
    )
    X = pd.DataFrame(
        np.random.randn(n, 1),
        index=idx,
        columns=["regressor"],
    )

    forecaster = AutoTS(
        model_list="superfast",
        max_generations=1,
        num_validations=0,
        random_seed=42,
    )

    fh = [1, 2, 3]
    forecaster.fit(y, X=X, fh=fh)

    future_idx = pd.date_range("2020-04-11", periods=3, freq="D")
    X_pred = pd.DataFrame(
        np.random.randn(3, 1),
        index=future_idx,
        columns=["regressor"],
    )

    y_pred = forecaster.predict(fh=fh, X=X_pred)
    assert y_pred.shape == (3, 2)
    assert list(y_pred.columns) == ["series_a", "series_b"]

    quantiles = forecaster.predict_quantiles(alpha=[0.1, 0.9])
    assert quantiles.shape == (3, 4)
