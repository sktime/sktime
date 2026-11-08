# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Training-window stride tests for the make_reduction strategies."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from sktime.forecasting.compose import make_reduction


class _RecordingRegressor(RegressorMixin, BaseEstimator):
    """Record exactly which rows are fitted, accepting 2D and 3D features."""

    def fit(self, X, y):
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.mean_ = np.asarray(y).mean(axis=0)
        return self

    def predict(self, X):
        if np.ndim(self.mean_) == 0:
            return np.full(len(X), self.mean_)
        return np.tile(self.mean_, (len(X), 1))


def _models(forecaster):
    if hasattr(forecaster, "estimators_"):
        return forecaster.estimators_
    return [forecaster.estimator_]


@pytest.mark.parametrize("strategy", ["direct", "recursive", "multioutput", "dirrec"])
@pytest.mark.parametrize("scitype", ["tabular-regressor", "time-series-regressor"])
@pytest.mark.parametrize("stride", [1, 2, 5, 100, np.int64(3)])
def test_reduction_stride_training_rows(strategy, scitype, stride):
    """All strategies fit matching strided feature/target rows and retain the fh."""
    y = pd.Series(np.arange(30.0))
    X = pd.DataFrame({"x": np.arange(30.0) + 100})
    X_future = pd.DataFrame({"x": [130.0, 131.0, 132.0]}, index=[30, 31, 32])
    params = dict(
        estimator=_RecordingRegressor(),
        strategy=strategy,
        scitype=scitype,
        window_length=4,
    )
    full = make_reduction(**params).fit(y, X=X, fh=[1, 3])
    sampled = make_reduction(**params, window_stride=stride).fit(y, X=X, fh=[1, 3])
    assert sampled.clone().get_params()["window_stride"] == stride
    for baseline, actual in zip(_models(full), _models(sampled)):
        np.testing.assert_array_equal(actual.X_, baseline.X_[::stride])
        np.testing.assert_array_equal(actual.y_, baseline.y_[::stride])
    pred = sampled.predict(fh=[1, 3], X=X_future)
    np.testing.assert_array_equal(pred.index, [30, 32])
    if stride == 1:
        pd.testing.assert_series_equal(pred, full.predict(fh=[1, 3], X=X_future))


@pytest.mark.parametrize("stride", [2, 3, 100])
def test_direct_stride_different_windows(stride):
    """Each horizon is truncated before applying stride to avoid padded targets."""
    y = pd.Series(np.arange(19.0))
    params = dict(
        estimator=_RecordingRegressor(),
        strategy="direct",
        window_length=4,
        windows_identical=False,
    )
    full = make_reduction(**params).fit(y, fh=[1, 4])
    sampled = make_reduction(**params, window_stride=stride).fit(y, fh=[1, 4])
    for baseline, actual in zip(_models(full), _models(sampled)):
        np.testing.assert_array_equal(actual.X_, baseline.X_[::stride])
        np.testing.assert_array_equal(actual.y_, baseline.y_[::stride])


@pytest.mark.parametrize("strategy", ["direct", "recursive"])
@pytest.mark.parametrize("stride", [2, 3])
@pytest.mark.parametrize("levels", [1, 2])
@pytest.mark.parametrize("custom_transformer", [False, True])
def test_global_stride_restarts_per_series(
    strategy, stride, levels, custom_transformer
):
    """Unequal pooled series retain their own first window and aligned targets."""
    frames = []
    for i, length in enumerate([20, 23, 26]):
        frame = pd.DataFrame({"y": np.arange(length, dtype=float) + i * 100})
        frame.index = pd.MultiIndex.from_tuples(
            [
                (f"series_{i}", t) if levels == 1 else ("group", f"series_{i}", t)
                for t in range(length)
            ]
        )
        frames.append(frame)
    y = pd.concat(frames)
    params = dict(
        estimator=_RecordingRegressor(),
        strategy=strategy,
        window_length=4,
        pooling="global",
    )
    if custom_transformer:
        from sktime.transformations.summarize import WindowSummarizer

        params.update(
            window_length=None,
            transformers=[WindowSummarizer(lag_feature={"lag": [1, 2, 3, 4]})],
        )
    X = y.rename(columns={"y": "exog"}) + 1000
    full = make_reduction(**params).fit(y, X=X, fh=[1, 3])
    sampled = make_reduction(**params, window_stride=stride).fit(y, X=X, fh=[1, 3])
    for baseline, actual in zip(_models(full), _models(sampled)):
        group_levels = list(range(baseline.X_.index.nlevels - 1))
        counts = baseline.X_.groupby(level=group_levels, sort=False).cumcount()
        rows = np.flatnonzero(counts.to_numpy() % stride == 0)
        pd.testing.assert_frame_equal(actual.X_, baseline.X_.iloc[rows])
        expected_y = (
            baseline.y_.iloc[rows]
            if isinstance(baseline.y_, pd.DataFrame)
            else baseline.y_[rows]
        )
        np.testing.assert_array_equal(actual.y_, expected_y)


def test_stride_explicit_windows():
    """Stride 3 keeps window origins 0, 3, 6, with the corresponding future labels."""
    f = make_reduction(
        _RecordingRegressor(), strategy="direct", window_length=3, window_stride=3
    ).fit(pd.Series(np.arange(12.0)), fh=[1, 3])
    expected_X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    for model, targets in zip(f.estimators_, [[3, 6, 9], [5, 8, 11]]):
        np.testing.assert_array_equal(model.X_, expected_X)
        np.testing.assert_array_equal(model.y_, targets)


@pytest.mark.parametrize("strategy", ["direct", "recursive", "multioutput", "dirrec"])
@pytest.mark.parametrize("stride", [0, -1, 1.5, True, np.bool_(False), None, "2"])
def test_invalid_stride(strategy, stride):
    """Reject nonpositive, nonintegral and boolean stride values at fit."""
    forecaster = make_reduction(
        _RecordingRegressor(), strategy=strategy, window_length=3, window_stride=stride
    )
    with pytest.raises(ValueError, match="window_stride must be a positive integer"):
        forecaster.fit(pd.Series(np.arange(20.0)), fh=[1, 3])
