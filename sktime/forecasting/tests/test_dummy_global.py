"""Tests for DummyGlobalForecaster."""

__author__ = ["SimonBlanke"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.dummy_global import DummyGlobalForecaster
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series


class TestDummyGlobalForecaster:
    """Unit tests for DummyGlobalForecaster."""

    def test_pretrain_fit_predict_mean_strategy(self):
        """Test basic pretrain-fit-predict workflow with mean strategy."""
        # Create panel data for pretraining
        y_panel = _make_hierarchical(
            hierarchy_levels=(3,), min_timepoints=10, max_timepoints=10, n_columns=1
        )

        forecaster = DummyGlobalForecaster(strategy="mean")
        forecaster.pretrain(y_panel)

        # Check that global mean was computed
        assert hasattr(forecaster, "global_mean_")
        assert hasattr(forecaster, "n_pretrain_instances_")
        assert forecaster.n_pretrain_instances_ == 3

        # Fit on a single series
        y_train = _make_series(n_columns=1, n_timepoints=20)
        forecaster.fit(y_train, fh=[1, 2, 3])
        y_pred = forecaster.predict()

        # Check predictions
        assert len(y_pred) == 3
        assert isinstance(y_pred, pd.Series)
        # All predictions should be the global mean
        np.testing.assert_array_almost_equal(
            y_pred.values, np.repeat(forecaster.global_mean_, 3)
        )

    def test_pretrain_fit_predict_last_strategy(self):
        """Test basic pretrain-fit-predict workflow with last strategy."""
        # Create panel data for pretraining
        y_panel = _make_hierarchical(
            hierarchy_levels=(2,), min_timepoints=10, max_timepoints=10, n_columns=1
        )

        forecaster = DummyGlobalForecaster(strategy="last")
        forecaster.pretrain(y_panel)

        # Fit on a single series
        y_train = _make_series(n_columns=1, n_timepoints=20)
        forecaster.fit(y_train, fh=[1, 2, 3])
        y_pred = forecaster.predict()

        # Check predictions
        assert len(y_pred) == 3
        assert isinstance(y_pred, pd.Series)
        # All predictions should be the last value
        expected_value = y_train.iloc[-1]
        np.testing.assert_array_almost_equal(
            y_pred.values, np.repeat(expected_value, 3)
        )

    def test_without_pretraining(self):
        """Test that forecaster works without pretraining (uses series mean)."""
        forecaster = DummyGlobalForecaster(strategy="mean")

        # Fit without pretraining
        y_train = _make_series(n_columns=1, n_timepoints=20)
        forecaster.fit(y_train, fh=[1, 2, 3])

        # Check that global_mean_ was computed from the training series
        assert hasattr(forecaster, "global_mean_")
        expected_mean = y_train.mean()
        np.testing.assert_almost_equal(forecaster.global_mean_, expected_mean)

        y_pred = forecaster.predict()

        assert len(y_pred) == 3
        np.testing.assert_array_almost_equal(
            y_pred.values, np.repeat(forecaster.global_mean_, 3)
        )
