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

    def test_pretrain_updates_state(self):
        """Test that pretrain() updates forecaster state correctly."""
        forecaster = DummyGlobalForecaster()

        assert forecaster.state == "new"

        y_panel = _make_hierarchical(
            hierarchy_levels=(2,), min_timepoints=10, max_timepoints=10
        )
        forecaster.pretrain(y_panel)

        assert forecaster.state == "pretrained"

        y_train = _make_series(n_columns=1, n_timepoints=20)
        forecaster.fit(y_train, fh=[1, 2, 3])

        assert forecaster.state == "fitted"

    def test_incremental_pretraining(self):
        """Test that pretrain can be called multiple times (incremental)."""
        forecaster = DummyGlobalForecaster()

        # First pretrain batch
        y_panel1 = _make_hierarchical(
            hierarchy_levels=(2,), min_timepoints=10, max_timepoints=10
        )
        forecaster.pretrain(y_panel1)
        mean_after_first = forecaster.global_mean_

        # Second pretrain batch
        y_panel2 = _make_hierarchical(
            hierarchy_levels=(3,), min_timepoints=10, max_timepoints=10
        )
        forecaster.pretrain(y_panel2)
        mean_after_second = forecaster.global_mean_

        # Mean should be updated (not necessarily equal due to different data)
        # Just check that it's computed and is a finite number
        assert np.isfinite(mean_after_second)
        assert forecaster.state == "pretrained"

    def test_multivariate_forecasting(self):
        """Test DummyGlobalForecaster with multivariate data."""
        # Create multivariate panel data
        y_panel = _make_hierarchical(
            hierarchy_levels=(3,), min_timepoints=10, max_timepoints=10, n_columns=2
        )
        forecaster = DummyGlobalForecaster(strategy="mean")
        forecaster.pretrain(y_panel)

        # Fit on multivariate series
        y_train = _make_series(n_columns=2, n_timepoints=20)
        forecaster.fit(y_train, fh=[1, 2, 3])
        y_pred = forecaster.predict()

        assert len(y_pred) == 3
        assert isinstance(y_pred, pd.DataFrame)
        assert y_pred.shape == (3, 2)  # 3 time points, 2 columns

    def test_get_pretrained_params(self):
        """Test get_pretrained_params method."""
        forecaster = DummyGlobalForecaster()

        # Before pretraining, should return empty dict
        params = forecaster.get_pretrained_params()
        assert params == {}

        # After pretraining
        y_panel = _make_hierarchical(
            hierarchy_levels=(2,), min_timepoints=10, max_timepoints=10
        )
        forecaster.pretrain(y_panel)
        params = forecaster.get_pretrained_params()

        # Should have pretrained attributes
        assert "global_mean_" in params
        assert "global_std_" in params
        assert "n_pretrain_instances_" in params
        assert "n_pretrain_timepoints_" in params

        # After fit, pretrained params should still be available
        y_train = _make_series(n_columns=1, n_timepoints=20)
        forecaster.fit(y_train, fh=[1, 2, 3])

        params_after_fit = forecaster.get_pretrained_params()
        # All pretrained params should still exist
        assert "global_mean_" in params_after_fit
        assert "global_std_" in params_after_fit

    def test_capability_tag(self):
        """Test that DummyGlobalForecaster has correct capability tag."""
        forecaster = DummyGlobalForecaster()

        # Should have capability:pretrain tag set to True
        assert forecaster.get_tag("capability:pretrain") is True
