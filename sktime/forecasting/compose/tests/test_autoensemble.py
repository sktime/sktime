#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of AutoEnsembleForecaster functionality."""

__author__ = ["mloning", "GuzalBulatova", "aiwalter", "RNKuhns", "AnH0ang"]

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sktime.datasets import load_longley
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.compose import (
    AutoEnsembleForecaster,
    RecursiveTabularRegressionForecaster,
)
from sktime.split import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.split.base import BaseSplitter
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(AutoEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


class RecordingWeightRegressor(RegressorMixin, BaseEstimator):
    """Regressor that records fit data and exposes deterministic coefficients."""

    def __init__(self, coef=None):
        self.coef = coef

    def fit(self, X, y):
        """Record meta-features and targets."""
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        n_features = self.X_.shape[1]

        if self.coef is None:
            self.coef_ = np.ones(n_features)
        else:
            self.coef_ = np.asarray(self.coef)

        return self

    def predict(self, X):
        """Predict a deterministic weighted sum."""
        return np.asarray(X) @ self.coef_


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


@pytest.mark.parametrize(
    "forecasters",
    [
        [
            (
                "dt",
                RecursiveTabularRegressionForecaster(
                    DecisionTreeRegressor(random_state=42), window_length=3
                ),
            ),
            (
                "lr",
                RecursiveTabularRegressionForecaster(
                    LinearRegression(), window_length=3
                ),
            ),
        ],
    ],
)
@pytest.mark.parametrize(
    "method",
    ["inverse-variance", "feature-importance"],
)
def test_autoensembler(forecasters, method):
    """Check that the prediction is a weighted mean of the individual predictions."""
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

    fh_test = ForecastingHorizon(y_test.index, is_relative=False)

    ensemble_forecaster = AutoEnsembleForecaster(forecasters=forecasters, method=method)
    ensemble_forecaster.fit(y_train, X_train)
    y_pred = ensemble_forecaster.predict(fh=fh_test, X=X_test)

    predictions = []
    for _, forecaster in forecasters:
        f = forecaster
        f.fit(y_train, X_train)
        f_pred = f.predict(fh=fh_test, X=X_test)
        predictions.append(f_pred)
    predictions = pd.DataFrame(predictions).T

    assert (predictions.min(axis=1) <= y_pred).all()
    assert (predictions.max(axis=1) >= y_pred).all()


def test_autoensemble_forecaster_accepts_cv_parameter():
    """Test that cv is accepted and exposed as an estimator parameter."""
    cv = ExpandingWindowSplitter(fh=[1, 2], initial_window=8, step_length=4)
    forecaster = AutoEnsembleForecaster(
        forecasters=_forecasters(), regressor=RecordingWeightRegressor(), cv=cv
    )

    assert forecaster.cv is cv
    assert forecaster.get_params()["cv"] is cv


@pytest.mark.parametrize(
    "cv",
    [
        ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=8, step_length=5),
        SlidingWindowSplitter(fh=[1, 2, 3], window_length=8, step_length=5),
    ],
)
def test_autoensemble_feature_importance_uses_temporal_cv(cv):
    """Test that feature-importance weights are learned from temporal folds."""
    y = _make_y()
    expected_X, expected_y = _expected_meta_data(y, cv)

    forecaster = AutoEnsembleForecaster(
        forecasters=_forecasters(),
        regressor=RecordingWeightRegressor(coef=[0.25, 0.75]),
        cv=cv,
    )
    forecaster.fit(y, fh=[1, 2, 3])

    np.testing.assert_array_equal(forecaster.regressor_.X_, expected_X)
    np.testing.assert_array_equal(forecaster.regressor_.y_, expected_y)
    np.testing.assert_array_equal(forecaster.weights_, [0.25, 0.75])

    for _, fitted_forecaster in forecaster.forecasters_:
        assert fitted_forecaster.n_obs_ == len(y)
        pd.testing.assert_index_equal(fitted_forecaster.fit_index_, y.index)

    y_pred = forecaster.predict()
    assert isinstance(y_pred, pd.Series)
    pd.testing.assert_index_equal(y_pred.index, pd.Index([28, 29, 30]))


def test_autoensemble_uses_positional_fh_with_gapped_integer_index():
    """Test integer fh validation with non-consecutive index labels."""
    y = _make_y()
    y.index = pd.Index(np.arange(len(y)) * 2)
    fh = [1, 2, 3]
    cv = ExpandingWindowSplitter(fh=fh, initial_window=8, step_length=5)

    forecaster = AutoEnsembleForecaster(
        forecasters=_forecasters(),
        regressor=RecordingWeightRegressor(coef=[0.25, 0.75]),
        cv=cv,
    )
    forecaster.fit(y, fh=fh)

    expected_X, expected_y = _expected_meta_data(y, cv)
    np.testing.assert_array_equal(forecaster.regressor_.X_, expected_X)
    np.testing.assert_array_equal(forecaster.regressor_.y_, expected_y)


@pytest.mark.parametrize(
    "cv",
    [
        ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=8, step_length=5),
        SlidingWindowSplitter(fh=[1, 2, 3], window_length=8, step_length=5),
    ],
)
def test_autoensemble_inverse_variance_uses_temporal_cv(cv):
    """Test that inverse-variance weights are estimated across all cv folds."""
    y = _make_y()
    expected_X, expected_y = _expected_meta_data(y, cv)
    errors = expected_y.reshape(-1, 1) - expected_X
    inv_var = 1 / np.var(errors, axis=0)
    expected_weights = inv_var / np.sum(inv_var)

    forecaster = AutoEnsembleForecaster(
        forecasters=_forecasters(), method="inverse-variance", cv=cv
    )
    forecaster.fit(y)

    np.testing.assert_allclose(forecaster.weights_, expected_weights)


def test_autoensemble_forecaster_raises_for_invalid_cv():
    """Test that non-sktime splitters are rejected clearly."""
    forecaster = AutoEnsembleForecaster(
        forecasters=_forecasters(), regressor=RecordingWeightRegressor(), cv=object()
    )

    with pytest.raises(TypeError, match="`cv` is not an instance"):
        forecaster.fit(_make_y())


def test_autoensemble_forecaster_raises_for_empty_cv():
    """Test that splitters producing no folds are rejected clearly."""
    forecaster = AutoEnsembleForecaster(
        forecasters=_forecasters(),
        regressor=RecordingWeightRegressor(),
        cv=EmptySplitter(),
    )

    with pytest.raises(ValueError, match="does not produce any valid"):
        forecaster.fit(_make_y())


def test_autoensemble_forecaster_raises_for_incompatible_cv_fh():
    """Test that cv test windows must match an explicit fit forecasting horizon."""
    cv = ExpandingWindowSplitter(fh=[1, 2], initial_window=8, step_length=5)
    forecaster = AutoEnsembleForecaster(
        forecasters=_forecasters(), regressor=RecordingWeightRegressor(), cv=cv
    )

    with pytest.raises(ValueError, match="forecasting horizon is incompatible"):
        forecaster.fit(_make_y(), fh=[1, 2, 3])


def test_autoensemble_forecaster_raises_for_non_temporal_cv_split():
    """Test that cv folds cannot train on observations they validate on."""
    forecaster = AutoEnsembleForecaster(
        forecasters=_forecasters(),
        regressor=RecordingWeightRegressor(),
        cv=InSampleSplitter(),
    )

    with pytest.raises(ValueError, match="temporal validation splits"):
        forecaster.fit(_make_y())
