#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of StackingForecaster functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import StackingForecaster
from sktime.split import (
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
)
from sktime.split.base import BaseSplitter
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(StackingForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """Regressor that records fit data and predicts row means."""

    def fit(self, X, y):
        """Record meta-features and targets."""
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        """Predict row means for deterministic point forecasts."""
        return np.asarray(X).mean(axis=1)


class TrainLengthForecaster(BaseForecaster):
    """Forecaster that predicts the number of observations seen in fit."""

    _tags = {
        "capability:exogenous": True,
        "capability:missing_values": True,
        "requires-fh-in-fit": False,
    }

    def __init__(self, offset=0):
        self.offset = offset
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Record training-window length and index."""
        self.n_obs_ = len(y)
        self.fit_index_ = y.index.copy()
        return self

    def _predict(self, fh=None, X=None):
        """Return the recorded training-window length plus offset."""
        index = fh.to_absolute_index(self.cutoff)
        y_pred = np.repeat(self.n_obs_ + self.offset, len(index))
        return pd.Series(y_pred, index=index)


class EmptySplitter(BaseSplitter):
    """Temporal splitter that yields no train/test splits."""

    def __init__(self, fh=1):
        super().__init__(fh=fh, window_length=1)

    def _split(self, y):
        """Yield no splits."""
        yield from ()


class InSampleSplitter(BaseSplitter):
    """Temporal splitter that produces a leaking in-sample test window."""

    def __init__(self):
        super().__init__(fh=[0], window_length=3)

    def _split(self, y):
        """Yield one split whose test window overlaps the train window."""
        yield np.array([0, 1, 2]), np.array([2])


class DuplicateWindowSplitter(BaseSplitter):
    """Temporal splitter that produces a duplicate positional index."""

    def __init__(self):
        super().__init__(fh=[1], window_length=3)

    def _split(self, y):
        """Yield one split whose train window contains a duplicate."""
        yield np.array([0, 1, 1]), np.array([3])


class OutOfBoundsSplitter(BaseSplitter):
    """Temporal splitter that produces an out-of-bounds positional index."""

    def __init__(self):
        super().__init__(fh=[1], window_length=3)

    def _split(self, y):
        """Yield one split whose test window is outside y."""
        yield np.array([0, 1, 2]), np.array([len(y)])


def _make_y(n_timepoints=28):
    return pd.Series(
        np.arange(n_timepoints, dtype=float),
        index=pd.RangeIndex(n_timepoints),
        name="y",
    )


def _forecasters():
    return [
        ("short", TrainLengthForecaster()),
        ("long", TrainLengthForecaster(offset=100)),
    ]


def _expected_meta_data(y, cv):
    X_meta = []
    y_meta = []

    for train_window, test_window in cv.split(y):
        X_meta.extend([[len(train_window), len(train_window) + 100]] * len(test_window))
        y_meta.extend(y.iloc[test_window].values)

    return np.asarray(X_meta), np.asarray(y_meta)


def test_stacking_forecaster_accepts_cv_parameter():
    """Test that cv is accepted and exposed as an estimator parameter."""
    cv = ExpandingWindowSplitter(fh=[1, 2], initial_window=8, step_length=4)
    forecaster = StackingForecaster(
        forecasters=_forecasters(), regressor=RecordingRegressor(), cv=cv
    )

    assert forecaster.cv is cv
    assert forecaster.get_params()["cv"] is cv


def test_stacking_forecaster_cv_none_matches_single_window_splitter():
    """Test that cv=None preserves the existing single-window behavior."""
    y = _make_y()
    fh = [1, 2, 3]

    forecaster_default = StackingForecaster(
        forecasters=_forecasters(), regressor=RecordingRegressor()
    )
    forecaster_default.fit(y, fh=fh)

    forecaster_single = StackingForecaster(
        forecasters=_forecasters(),
        regressor=RecordingRegressor(),
        cv=SingleWindowSplitter(fh=fh),
    )
    forecaster_single.fit(y, fh=fh)

    np.testing.assert_array_equal(
        forecaster_default.regressor_.X_, forecaster_single.regressor_.X_
    )
    np.testing.assert_array_equal(
        forecaster_default.regressor_.y_, forecaster_single.regressor_.y_
    )


@pytest.mark.parametrize(
    "cv",
    [
        ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=8, step_length=5),
        SlidingWindowSplitter(fh=[1, 2, 3], window_length=8, step_length=5),
    ],
)
def test_stacking_forecaster_trains_meta_regressor_from_temporal_cv(cv):
    """Test that meta-regressor data are temporally out-of-fold predictions."""
    y = _make_y()
    expected_X, expected_y = _expected_meta_data(y, cv)

    forecaster = StackingForecaster(
        forecasters=_forecasters(), regressor=RecordingRegressor(), cv=cv
    )
    forecaster.fit(y, fh=[1, 2, 3])

    np.testing.assert_array_equal(forecaster.regressor_.X_, expected_X)
    np.testing.assert_array_equal(forecaster.regressor_.y_, expected_y)

    for _, fitted_forecaster in forecaster.forecasters_:
        assert fitted_forecaster.n_obs_ == len(y)
        pd.testing.assert_index_equal(fitted_forecaster.fit_index_, y.index)


def test_stacking_forecaster_preserves_prediction_index_and_type():
    """Test point forecast output type and multi-step prediction index."""
    y = _make_y()
    fh = [1, 2, 4]
    cv = ExpandingWindowSplitter(fh=fh, initial_window=8, step_length=5)

    forecaster = StackingForecaster(
        forecasters=_forecasters(), regressor=RecordingRegressor(), cv=cv
    )
    forecaster.fit(y, fh=fh)
    y_pred = forecaster.predict()

    assert isinstance(y_pred, pd.Series)
    assert y_pred.name == y.name
    pd.testing.assert_index_equal(y_pred.index, pd.Index([28, 29, 31]))


def test_stacking_forecaster_uses_positional_fh_with_gapped_integer_index():
    """Test integer fh validation with non-consecutive index labels."""
    y = _make_y()
    y.index = pd.Index(np.arange(len(y)) * 2)
    fh = [1, 2, 3]
    cv = ExpandingWindowSplitter(fh=fh, initial_window=8, step_length=5)

    forecaster = StackingForecaster(
        forecasters=_forecasters(), regressor=RecordingRegressor(), cv=cv
    )
    forecaster.fit(y, fh=fh)

    expected_X, expected_y = _expected_meta_data(y, cv)
    np.testing.assert_array_equal(forecaster.regressor_.X_, expected_X)
    np.testing.assert_array_equal(forecaster.regressor_.y_, expected_y)


def test_stacking_forecaster_is_deterministic_with_deterministic_components():
    """Test deterministic predictions with deterministic base and meta estimators."""
    y = _make_y()
    fh = [1, 2, 3]
    cv = ExpandingWindowSplitter(fh=fh, initial_window=8, step_length=5)

    forecaster_1 = StackingForecaster(
        forecasters=_forecasters(), regressor=RecordingRegressor(), cv=cv
    )
    forecaster_2 = StackingForecaster(
        forecasters=_forecasters(), regressor=RecordingRegressor(), cv=cv
    )

    forecaster_1.fit(y, fh=fh)
    forecaster_2.fit(y, fh=fh)

    pd.testing.assert_series_equal(forecaster_1.predict(), forecaster_2.predict())


def test_stacking_forecaster_raises_for_invalid_cv():
    """Test that non-sktime splitters are rejected clearly."""
    forecaster = StackingForecaster(
        forecasters=_forecasters(), regressor=RecordingRegressor(), cv=object()
    )

    with pytest.raises(TypeError, match="`cv` is not an instance"):
        forecaster.fit(_make_y(), fh=[1, 2, 3])


def test_stacking_forecaster_raises_for_empty_cv():
    """Test that splitters producing no folds are rejected clearly."""
    forecaster = StackingForecaster(
        forecasters=_forecasters(), regressor=RecordingRegressor(), cv=EmptySplitter()
    )

    with pytest.raises(ValueError, match="does not produce any valid"):
        forecaster.fit(_make_y(), fh=[1])


def test_stacking_forecaster_raises_for_incompatible_cv_fh():
    """Test that cv test windows must match the fit forecasting horizon."""
    cv = ExpandingWindowSplitter(fh=[1, 2], initial_window=8, step_length=5)
    forecaster = StackingForecaster(
        forecasters=_forecasters(), regressor=RecordingRegressor(), cv=cv
    )

    with pytest.raises(ValueError, match="forecasting horizon is incompatible"):
        forecaster.fit(_make_y(), fh=[1, 2, 3])


def test_stacking_forecaster_raises_for_non_temporal_cv_split():
    """Test that cv folds cannot train on observations they validate on."""
    forecaster = StackingForecaster(
        forecasters=_forecasters(),
        regressor=RecordingRegressor(),
        cv=InSampleSplitter(),
    )

    with pytest.raises(ValueError, match="temporal validation splits"):
        forecaster.fit(_make_y(), fh=[1])


def test_stacking_forecaster_raises_for_duplicate_cv_window_indices():
    """Test that cv train/test windows must be duplicate-free."""
    forecaster = StackingForecaster(
        forecasters=_forecasters(),
        regressor=RecordingRegressor(),
        cv=DuplicateWindowSplitter(),
    )

    with pytest.raises(ValueError, match="duplicate-free"):
        forecaster.fit(_make_y(), fh=[1])


def test_stacking_forecaster_raises_for_out_of_bounds_cv_window_indices():
    """Test that cv train/test windows must stay within y bounds."""
    forecaster = StackingForecaster(
        forecasters=_forecasters(),
        regressor=RecordingRegressor(),
        cv=OutOfBoundsSplitter(),
    )

    with pytest.raises(ValueError, match="within the bounds"):
        forecaster.fit(_make_y(), fh=[1])
