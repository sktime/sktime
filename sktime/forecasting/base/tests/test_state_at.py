"""Tests for state-aware private BaseForecaster methods."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["SimonBlanke"]

import numpy as np
import pytest

from sktime.forecasting.dummy_global import DummyGlobalForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series


def _make_panel():
    """Make small panel data for pretraining."""
    return _make_hierarchical(
        hierarchy_levels=(3,), min_timepoints=10, max_timepoints=10, n_columns=1
    )


def _make_pretrained_dummy(strategy="mean"):
    """Return a pretrained DummyGlobalForecaster."""
    forecaster = DummyGlobalForecaster(strategy=strategy)
    forecaster.pretrain(_make_panel())
    return forecaster


class TestResetAt:
    """Tests for _reset_at."""

    def test_pretrained_keeps_pretrained_attrs_removes_fitted_attrs(self):
        """Reset to pretrained keeps pretraining state, but removes fit state."""
        forecaster = _make_pretrained_dummy(strategy="last")
        pretrain_mean = forecaster.global_mean_

        forecaster.fit(_make_series(n_timepoints=20), fh=[1, 2, 3])
        assert hasattr(forecaster, "last_value_")

        forecaster._reset_at("pretrained")

        assert forecaster.state == "pretrained"
        assert hasattr(forecaster, "global_mean_")
        assert not hasattr(forecaster, "last_value_")
        np.testing.assert_almost_equal(forecaster.global_mean_, pretrain_mean)

    def test_new_discards_pretrained_attrs(self):
        """Reset to new behaves like the public reset."""
        forecaster = _make_pretrained_dummy()

        forecaster._reset_at("new")

        assert forecaster.state == "new"
        assert not hasattr(forecaster, "global_mean_")
        assert not hasattr(forecaster, "_pretrained_attrs")

    def test_pretrained_without_pretraining_degrades_to_new(self):
        """Target state is an upper bound, not a state guarantee."""
        forecaster = DummyGlobalForecaster(strategy="last")
        forecaster.fit(_make_series(n_timepoints=20), fh=[1, 2, 3])

        forecaster._reset_at("pretrained")

        assert forecaster.state == "new"
        assert not hasattr(forecaster, "last_value_")

    def test_no_pretrain_capability_falls_back_to_reset(self):
        """Forecasters without pretrain capability use ordinary reset."""
        forecaster = NaiveForecaster()
        forecaster.fit(_make_series(n_timepoints=20), fh=[1, 2, 3])

        forecaster._reset_at("pretrained")

        assert not forecaster.is_fitted

    def test_invalid_state_raises(self):
        """Only new and pretrained are valid target states."""
        with pytest.raises(ValueError, match="target state"):
            DummyGlobalForecaster()._reset_at("fitted")
        with pytest.raises(ValueError, match="target state"):
            NaiveForecaster()._reset_at("bogus")

    def test_pretrain_fitted_params_tag_is_source_of_truth(self):
        """A non-empty pretrain:fitted_params tag selects protected attrs."""
        forecaster = _make_pretrained_dummy()
        forecaster.set_tags(**{"pretrain:fitted_params": ["global_mean_"]})

        forecaster._reset_at("pretrained")

        assert forecaster.state == "pretrained"
        assert hasattr(forecaster, "global_mean_")
        assert not hasattr(forecaster, "global_std_")

    def test_empty_pretrain_fitted_params_tag_uses_runtime_attrs(self):
        """An empty tag falls back to the runtime _pretrained_attrs list."""
        forecaster = _make_pretrained_dummy()
        forecaster.set_tags(**{"pretrain:fitted_params": []})

        forecaster._reset_at("pretrained")

        assert forecaster.state == "pretrained"
        assert hasattr(forecaster, "global_mean_")
        assert hasattr(forecaster, "global_std_")
