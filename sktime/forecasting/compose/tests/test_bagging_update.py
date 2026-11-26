"""Regression tests for bagging updates and regenerated bootstrap histories."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.compose import BaggingForecaster, ForecastByLevel, YfromX
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.stream import UpdateRefitsEvery
from sktime.forecasting.trend import TrendForecaster
from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.bootstrap import MovingBlockBootstrapTransformer

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting.compose._bagging"),
    reason="run only if bagging module has changed (if incremental testing requested)",
)


def _make_bagging(forecaster, n_series=3):
    return BaggingForecaster(
        bootstrap_transformer=MovingBlockBootstrapTransformer(
            n_series=n_series,
            block_length=1,
            sampling_replacement=False,
            random_state=42,
        ),
        forecaster=forecaster,
    )


@pytest.mark.parametrize("period_index", [False, True])
@pytest.mark.parametrize("n_series", [1, 3])
@pytest.mark.parametrize("refit_wrapper", [False, True])
def test_bagging_repeated_updates(period_index, n_series, refit_wrapper):
    """Splitting observations across updates retains the full-history mean."""
    index = (
        pd.period_range("2020-01", periods=10, freq="M")
        if period_index
        else pd.RangeIndex(10)
    )
    y = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 101.0, 200.0, 201.0],
        index=index,
        name="y",
    )
    inner = NaiveForecaster(strategy="mean")
    if refit_wrapper:
        inner = ForecastByLevel(UpdateRefitsEvery(inner, refit_interval=0))
    one, split, fresh = [_make_bagging(inner, n_series) for _ in range(3)]
    one.fit(y.iloc[:6], fh=[1]).update(y.iloc[6:])
    split.fit(y.iloc[:6], fh=[1])
    split.update(y.iloc[6:8]).update(y.iloc[8:])
    fresh.fit(y, fh=[1])
    # Every bootstrap is a permutation, so randomness cannot change its mean.
    for model in [one, split, fresh]:
        np.testing.assert_allclose(model.predict(), y.mean())
        pd.testing.assert_frame_equal(model._cur_y, y.to_frame())
        assert model._y_bs_ix.get_level_values(-1).nunique() == len(y)


def test_bagging_exogenous_history():
    """Repeated updates replicate all exogenous rows for each bootstrap instance."""
    from sklearn.linear_model import LinearRegression

    index = pd.period_range("2020-01", periods=11, freq="M")
    X = pd.DataFrame({"x": np.arange(11, dtype=float)}, index=index)
    y = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 101.0, 200.0, 201.0],
        index=index[:10],
        name="y",
    )
    split = _make_bagging(YfromX(LinearRegression()))
    split.fit(y.iloc[:6], X=X.iloc[:6], fh=[1])
    split.update(y.iloc[6:8], X=X.iloc[6:8])
    split.update(y.iloc[8:], X=X.iloc[8:10])
    fresh = _make_bagging(YfromX(LinearRegression())).fit(y, X=X.iloc[:10], fh=[1])
    pd.testing.assert_series_equal(
        split.predict(X=X.iloc[10:]), fresh.predict(X=X.iloc[10:])
    )
    pd.testing.assert_frame_equal(split._cur_X, X.iloc[:10])
    pd.testing.assert_frame_equal(split._cur_y, y.to_frame())
    replicated = split._gen_X_bootstraps(split._cur_X)
    pd.testing.assert_index_equal(replicated.index, split._y_bs_ix, check_names=False)


def test_bagging_update_without_parameter_updates():
    """Frozen updates retain history for a later refit without changing coefficients."""
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 101.0, 200.0, 201.0], name="y")
    model = _make_bagging(TrendForecaster()).fit(y.iloc[:6], fh=[1])
    before = model.predict([5])
    model.update(y.iloc[6:8], update_params=False)
    pd.testing.assert_series_equal(model.predict([3]), before)
    pd.testing.assert_frame_equal(model._cur_y, y.iloc[:8].to_frame())
    model.update(y.iloc[8:])
    fresh = _make_bagging(TrendForecaster()).fit(y, fh=[1])
    pd.testing.assert_series_equal(model.predict([1]), fresh.predict())
