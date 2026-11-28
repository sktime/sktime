"""Tests the VAR model."""

__author__ = ["thayeylolu"]
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon

#
from sktime.forecasting.var import VAR
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(VAR),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("coverage", [[0.9], [0.9, 0.5]])
def test_var_interval_overlapping_column_names(coverage):
    """Each interval must belong to its target even if target names overlap."""
    from statsmodels.tsa.api import VAR as StatsmodelsVAR

    rng = np.random.default_rng(21)
    y = pd.DataFrame(
        {"a": rng.normal(size=48), "aa": rng.normal(20, 2, size=48)},
        index=pd.period_range("2020-01", periods=48, freq="M", name="time"),
    )
    fh = [2, 5]
    result = VAR(maxlags=1).fit(y).predict_interval(fh, coverage=coverage)
    reference = StatsmodelsVAR(y).fit(maxlags=1)

    expected_index = pd.PeriodIndex([y.index[-1] + step for step in fh], name="time")
    pd.testing.assert_index_equal(result.index, expected_index)
    columns = pd.MultiIndex.from_product([y.columns, coverage, ["lower", "upper"]])
    pd.testing.assert_index_equal(result.columns, columns)
    rows = np.array(fh) - 1
    for cov in coverage:
        _, lower, upper = reference.forecast_interval(
            y.to_numpy()[-1:], steps=max(fh), alpha=1 - cov
        )
        for j, name in enumerate(y.columns):
            assert_allclose(result[name, cov, "lower"], lower[rows, j])
            assert_allclose(result[name, cov, "upper"], upper[rows, j])


@pytest.mark.skipif(
    not run_test_for_class(VAR),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_VAR_against_statsmodels():
    """Compares Sktime's and Statsmodel's VAR."""
    from statsmodels.tsa.api import VAR as _VAR

    pandas2 = _check_soft_dependencies("pandas>=2.0.0", severity="none")
    if pandas2:
        freq = "ME"
    else:
        freq = "M"

    index = pd.date_range(start="2005", end="2006-12", freq=freq)
    df = pd.DataFrame(
        np.random.randint(0, 100, size=(23, 2)),
        columns=list("AB"),
        index=pd.PeriodIndex(index),
    )
    train, test = temporal_train_test_split(df)
    sktime_model = VAR()
    fh = ForecastingHorizon([1, 3, 4, 5, 7, 9])
    sktime_model.fit(train)
    y_pred = sktime_model.predict(fh=fh)

    stats = _VAR(train)
    stats_fit = stats.fit()
    fh_int = fh.to_relative(train.index[-1])
    lagged = stats_fit.k_ar
    y_pred_stats = stats_fit.forecast(train.values[-lagged:], steps=fh_int[-1])
    new_arr = []
    for i in fh_int:
        new_arr.append(y_pred_stats[i - 1])
    assert_allclose(y_pred, new_arr)
