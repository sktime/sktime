"""Regression tests for combined updates of stream policy wrappers."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.croston import Croston
from sktime.forecasting.stream import DontUpdate, UpdateEvery, UpdateRefitsEvery
from sktime.forecasting.trend import TrendForecaster
from sktime.tests.test_switch import run_test_module_changed

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting.stream"),
    reason="run only if stream module has changed (if incremental testing requested)",
)


def _make_wrapper(policy):
    if policy == "refit":
        return UpdateRefitsEvery(TrendForecaster(), refit_interval=2)
    if policy == "never":
        return DontUpdate(Croston(smoothing=0.5))
    interval = None if policy == "infinite" else 2
    return UpdateEvery(Croston(smoothing=0.5), update_interval=interval)


@pytest.mark.parametrize("policy", ["never", "infinite", "periodic", "refit"])
@pytest.mark.parametrize("index_kind", ["range", "period", "datetime"])
@pytest.mark.parametrize("update_params", [False, True])
@pytest.mark.parametrize("remember_data", [False, True])
def test_combined_update_honors_policy(
    policy, index_kind, update_params, remember_data
):
    """Combined calls match separate calls before, at and after a schedule boundary."""
    index = {
        "range": pd.RangeIndex(6),
        "period": pd.period_range("2020-01", periods=6, freq="M"),
        "datetime": pd.date_range("2020-01-01", periods=6, freq="D"),
    }[index_kind]
    y = pd.Series([1.0, 0.0, 1.0, 100.0, 101.0, 200.0], index=index, name="y")
    separate, combined = [
        _make_wrapper(policy)
        .set_config(remember_data=remember_data)
        .fit(y.iloc[:3], fh=[1])
        for _ in range(2)
    ]
    for i in range(3, 6):
        new_y = y.iloc[[i]]
        separate.update(new_y, update_params=update_params)
        expected = separate.predict()
        actual = combined.update_predict_single(new_y, update_params=update_params)
        pd.testing.assert_series_equal(actual, expected)
        pd.testing.assert_index_equal(combined.cutoff, separate.cutoff)
        pd.testing.assert_index_equal(
            combined.forecaster_.cutoff, separate.forecaster_.cutoff
        )
        if policy != "never":
            pd.testing.assert_series_equal(combined._y, y.iloc[: i + 1])
        if policy in ["never", "infinite"]:
            np.testing.assert_allclose(actual, 2 / 3)
    for attr in ["last_fit_cutoff_", "last_update_cutoff_"]:
        if hasattr(separate, attr):
            assert getattr(combined, attr) == getattr(separate, attr)


@pytest.mark.parametrize("policy", ["never", "infinite", "periodic", "refit"])
@pytest.mark.parametrize("reset_forecaster", [False, True])
def test_rolling_update_honors_policy(policy, reset_forecaster):
    """Rolling forecasts apply the wrapper policy at every cutoff."""
    y = pd.Series([1.0, 0.0, 1.0, 100.0, 101.0, 200.0], name="y")
    manual, rolling = [_make_wrapper(policy).fit(y.iloc[:3], fh=[1]) for _ in range(2)]
    expected = []
    for i in [3, 4]:
        # The default rolling splitter supplies expanding update windows.
        manual.update(y.iloc[3 : i + 1])
        expected.append(manual.predict())
    actual = rolling.update_predict(y.iloc[3:], reset_forecaster=reset_forecaster)
    pd.testing.assert_series_equal(actual, pd.concat(expected))
    assert rolling.cutoff[0] == (2 if reset_forecaster else 4)


def test_combined_refit_with_exogenous_history():
    """Combined refits pool target and exogenous history together."""
    from sklearn.linear_model import LinearRegression

    from sktime.forecasting.compose import YfromX

    y = pd.Series([1.0, 2.0, 3.0, 100.0, 101.0, 200.0], name="y")
    X = pd.DataFrame({"x": np.arange(7, dtype=float)})
    separate, combined = [
        UpdateRefitsEvery(YfromX(LinearRegression()), refit_interval=2).fit(
            y.iloc[:3], X=X.iloc[:3], fh=[0]
        )
        for _ in range(2)
    ]
    for i in range(3, 6):
        X_new = X.iloc[[i]]
        separate.update(y.iloc[[i]], X=X_new)
        expected = separate.predict(X=X_new)
        actual = combined.update_predict_single(y.iloc[[i]], X=X_new)
        pd.testing.assert_series_equal(actual, expected)
        pd.testing.assert_frame_equal(combined._X, separate._X)
        pd.testing.assert_series_equal(combined._y, y.iloc[: i + 1])
