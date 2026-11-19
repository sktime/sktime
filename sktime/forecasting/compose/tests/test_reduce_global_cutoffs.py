"""Regression tests for global recursive reduction with different series cutoffs."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose._reduce import _RecursiveReducer
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.summarize import WindowSummarizer

pytestmark = pytest.mark.skipif(
    not run_test_for_class(_RecursiveReducer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _make_panel(index_kind="integer", hierarchical=False, same_end=False):
    """Make linear series, including two sharing the earlier cutoff."""
    if index_kind == "integer":
        times = pd.Index(range(50))
    elif index_kind == "period":
        times = pd.period_range("2020-01", periods=50, freq="M")
    else:
        tz = "America/New_York" if index_kind == "timezone" else None
        times = pd.date_range("2024-03-01", periods=50, freq="D", tz=tz)

    # Deliberately put the instance keys out of sorted order.
    keys = [("west", "b"), ("east", "a"), ("west", "c")]
    names = ["region", "instance", "time"] if hierarchical else ["instance", "time"]
    pieces = []
    for i, (key, stop) in enumerate(zip(keys, [20, 30, 20])):
        if same_end:
            start, stop = i * 3, 30
        else:
            start = 0
        key = key if hierarchical else (key[-1],)
        index = pd.MultiIndex.from_tuples(
            [key + (t,) for t in times[start:stop]], names=names
        )
        pieces.append(
            pd.DataFrame(
                {"y": 100 * i + np.arange(start, stop, dtype=float)}, index=index
            )
        )
    return pd.concat(pieces), times


@pytest.mark.parametrize("index_kind", ["integer", "period", "datetime", "timezone"])
@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_global_recursive_per_instance_cutoffs(index_kind, hierarchical, absolute):
    """Each series uses its own observed tail and forecast dates (#5614)."""
    y, times = _make_panel(index_kind, hierarchical)
    fh = (
        ForecastingHorizon(times[[30, 32, 34]], is_relative=False)
        if absolute
        else ForecastingHorizon([1, 3, 5], is_relative=True)
    )
    forecaster = make_reduction(
        LinearRegression(), strategy="recursive", window_length=3, pooling="global"
    ).fit(y)
    cutoff = forecaster.cutoff.copy()
    history = forecaster._cur_y.copy(deep=True)
    fitted_regressor = forecaster.estimator_

    prediction = forecaster.predict(fh=fh)
    expected = []
    for i, (key, stop) in enumerate(zip(["b", "a", "c"], [20, 30, 20])):
        positions = [30, 32, 34] if absolute else [stop, stop + 2, stop + 4]
        instance = ("east" if key == "a" else "west", key) if hierarchical else (key,)
        index = pd.MultiIndex.from_tuples(
            [instance + (times[t],) for t in positions], names=y.index.names
        )
        expected.append(pd.DataFrame({"y": 100 * i + np.array(positions)}, index=index))

    assert_frame_equal(
        prediction, pd.concat(expected).sort_index(), check_dtype=False, atol=1e-10
    )
    assert_frame_equal(y, history)
    assert_frame_equal(forecaster._cur_y, history)
    assert_index_equal(forecaster.cutoff, cutoff)
    assert forecaster.estimator_ is fitted_regressor
    assert_frame_equal(forecaster.predict(fh=fh), prediction)


@pytest.mark.parametrize("hierarchical", [False, True])
def test_global_recursive_unequal_lengths_common_cutoff(hierarchical):
    """Different starts with a common cutoff retain the aligned prediction path."""
    y, _ = _make_panel(hierarchical=hierarchical, same_end=True)
    y = y.sort_index()
    forecaster = make_reduction(
        LinearRegression(), strategy="recursive", window_length=3, pooling="global"
    ).fit(y)
    prediction = forecaster.predict(fh=[1, 3])
    np.testing.assert_allclose(prediction["y"], [130, 132, 30, 32, 230, 232])


def test_global_recursive_cutoffs_after_update():
    """Prediction uses per-instance cutoffs refreshed by update without refitting."""
    y, _ = _make_panel()
    forecaster = make_reduction(
        LinearRegression(), strategy="recursive", window_length=3, pooling="global"
    ).fit(y)
    index = pd.MultiIndex.from_product([["b"], [20, 21]], names=y.index.names)
    update = pd.DataFrame({"y": [20.0, 21.0]}, index=index)
    forecaster.update(update, update_params=False)
    prediction = forecaster.predict(fh=[1, 3])
    expected = pd.DataFrame(
        {"y": [22.0, 24.0, 130.0, 132.0, 220.0, 222.0]},
        index=pd.MultiIndex.from_tuples(
            [("b", 22), ("b", 24), ("a", 30), ("a", 32), ("c", 20), ("c", 22)],
            names=y.index.names,
        ),
    )
    assert_frame_equal(prediction, expected.sort_index(), atol=1e-10)


def test_global_recursive_cutoffs_with_transformers():
    """Custom lag feature transformers also use the correct per-series windows."""
    y, _ = _make_panel()
    forecaster = make_reduction(
        LinearRegression(),
        strategy="recursive",
        window_length=None,
        transformers=[
            WindowSummarizer(lag_feature={"lag": [1]}, n_jobs=1),
            WindowSummarizer(lag_feature={"mean": [[1, 3]]}, n_jobs=1),
        ],
        pooling="global",
    ).fit(y)
    prediction = forecaster.predict(fh=[1, 3])
    expected_index = pd.MultiIndex.from_tuples(
        [("b", 20), ("b", 22), ("a", 30), ("a", 32), ("c", 20), ("c", 22)],
        names=y.index.names,
    )
    expected = pd.DataFrame(
        {"y": [20.0, 22.0, 130.0, 132.0, 220.0, 222.0]}, index=expected_index
    )
    assert_frame_equal(prediction, expected.sort_index(), atol=1e-10)


@pytest.mark.parametrize("absolute", [False, True])
def test_global_recursive_cutoffs_with_exogenous(absolute):
    """Prediction-time covariates are selected by each instance's actual date."""
    rng = np.random.default_rng(5614)
    histories, covariates, futures, expected = [], [], [], []
    for key, stop in [("b", 20), ("a", 30)]:
        x = rng.normal(size=35)
        values = [0.0]
        for t in range(1, len(x)):
            values.append(0.5 * values[-1] + 2 * x[t])
        index = pd.MultiIndex.from_product([[key], range(stop)], names=["id", "time"])
        histories.append(pd.DataFrame({"y": values[:stop]}, index=index))
        x_index = pd.MultiIndex.from_product([[key], range(35)], names=index.names)
        covariates.append(pd.DataFrame({"x": x[:stop]}, index=index))
        futures.append(pd.DataFrame({"x": x[stop:]}, index=x_index[stop:]))
        positions = [30, 32] if absolute else [stop, stop + 2]
        pred_index = pd.MultiIndex.from_product([[key], positions], names=index.names)
        expected.append(
            pd.DataFrame({"y": np.array(values)[positions]}, index=pred_index)
        )
    y, X = pd.concat(histories), pd.concat(covariates)
    forecaster = make_reduction(
        LinearRegression(), strategy="recursive", window_length=1, pooling="global"
    ).fit(y, X=X)
    fh = ForecastingHorizon([30, 32], is_relative=False) if absolute else [1, 3]
    prediction = forecaster.predict(fh=fh, X=pd.concat(futures))
    assert_frame_equal(prediction, pd.concat(expected).sort_index(), atol=1e-10)


def test_global_recursive_cutoffs_keep_pooled_fit(monkeypatch):
    """Grouping the prediction context must not fit separate per-instance models."""
    fit_calls = []
    fit = LinearRegression.fit

    def record_fit(self, X, y):
        fit_calls.append(len(y))
        return fit(self, X, y)

    monkeypatch.setattr(LinearRegression, "fit", record_fit)
    y, _ = _make_panel()
    forecaster = make_reduction(
        LinearRegression(), strategy="recursive", window_length=3, pooling="global"
    ).fit(y)
    forecaster.predict(fh=[1, 3])
    assert fit_calls == [len(y) - 3 * 3]
