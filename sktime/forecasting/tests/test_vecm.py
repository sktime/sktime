"""Tests the VAR model."""

__author__ = ["thayeylolu", "AurumnPegasus"]
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.vecm import VECM
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(VECM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("steps", [[1], [4], [1, 3, 5]])
@pytest.mark.parametrize("coverage", [[0.9], [0.9, 0.5]])
def test_vecm_prediction_intervals_against_statsmodels(steps, coverage):
    """Each endpoint belongs to its requested step, variable and coverage."""
    from statsmodels.tsa.vector_ar.vecm import VECM as StatsmodelsVECM

    rng = np.random.default_rng(21)
    walk = rng.normal(size=64).cumsum()
    y = pd.DataFrame(
        {"z": walk, "a": 18 + 2 * walk + rng.normal(size=64)},
        index=pd.period_range("2019-01", periods=64, freq="M"),
    )
    result = VECM().fit(y).predict_interval(fh=steps, coverage=coverage)
    reference = StatsmodelsVECM(y).fit()

    expected_index = pd.PeriodIndex([y.index[-1] + step for step in steps])
    pd.testing.assert_index_equal(result.index, expected_index)
    expected_columns = pd.MultiIndex.from_product(
        [y.columns, coverage, ["lower", "upper"]]
    )
    pd.testing.assert_index_equal(result.columns, expected_columns)
    for c in coverage:
        _, lower, upper = reference.predict(steps=max(steps), alpha=1 - c)
        for j, variable in enumerate(y.columns):
            assert_allclose(result[variable, c, "lower"], lower[np.array(steps) - 1, j])
            assert_allclose(result[variable, c, "upper"], upper[np.array(steps) - 1, j])


@pytest.mark.skipif(
    not run_test_for_class(VECM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_VECM_against_statsmodels():
    """Compares Sktime's and Statsmodel's VECM."""
    from statsmodels.tsa.api import VECM as _VECM

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
    sktime_model = VECM()
    fh = ForecastingHorizon([1, 3, 4, 5, 7, 9])
    _ = sktime_model.fit(train)
    y_pred = sktime_model.predict(fh=fh)

    stats = _VECM(train)
    stats_fit = stats.fit()
    fh_int = fh.to_relative(train.index[-1])
    # lagged = stats_fit.k_ar
    y_pred_stats = stats_fit.predict(steps=fh_int[-1])
    new_arr = []
    for i in fh_int:
        new_arr.append(y_pred_stats[i - 1])
    # print("predicted: \n")
    # print(y_pred)
    # print("actual: \n")
    # print(new_arr)
    assert_allclose(y_pred, new_arr)
