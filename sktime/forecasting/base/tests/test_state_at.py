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


class TestSetParamsAt:
    """Tests for _set_params_at."""

    def test_pretrained_keeps_pretrained_attrs(self):
        """Setting params at pretrained keeps pretraining state."""
        forecaster = _make_pretrained_dummy()
        pretrain_mean = forecaster.global_mean_

        forecaster._set_params_at("pretrained", {"strategy": "last"})

        assert forecaster.strategy == "last"
        assert forecaster.state == "pretrained"
        np.testing.assert_almost_equal(forecaster.global_mean_, pretrain_mean)

    def test_new_discards_pretrained_attrs(self):
        """Setting params at new discards pretraining state."""
        forecaster = _make_pretrained_dummy()

        forecaster._set_params_at("new", {"strategy": "last"})

        assert forecaster.strategy == "last"
        assert forecaster.state == "new"
        assert not hasattr(forecaster, "global_mean_")

    def test_empty_params_is_noop(self):
        """Empty params short-circuit without reset, matching set_params."""
        forecaster = _make_pretrained_dummy()

        forecaster._set_params_at("new", {})

        assert forecaster.state == "pretrained"
        assert hasattr(forecaster, "global_mean_")

    def test_no_pretrain_capability_behaves_like_set_params(self):
        """Without pretrain capability, state-aware set_params is ordinary."""
        forecaster = NaiveForecaster(strategy="last")

        forecaster._set_params_at("pretrained", {"strategy": "mean"})

        assert forecaster.strategy == "mean"

    def test_invalid_key_raises(self):
        """Unresolvable parameter keys raise, as in set_params."""
        with pytest.raises(ValueError, match="Invalid parameter keys"):
            DummyGlobalForecaster()._set_params_at("new", {"no_such_param": 1})

    def test_invalid_state_raises_before_setting_params(self):
        """Invalid states raise before mutating parameters."""
        forecaster = DummyGlobalForecaster(strategy="mean")

        with pytest.raises(ValueError, match="target state"):
            forecaster._set_params_at("fitted", {"strategy": "last"})

        assert forecaster.strategy == "mean"


class TestCloneAt:
    """Tests for _clone_at."""

    def test_pretrained_carries_pretrained_attrs(self):
        """Clone at pretrained carries pretraining state."""
        forecaster = _make_pretrained_dummy()
        pretrain_mean = forecaster.global_mean_

        cloned = forecaster._clone_at("pretrained")

        assert cloned is not forecaster
        assert cloned.state == "pretrained"
        np.testing.assert_almost_equal(cloned.global_mean_, pretrain_mean)

    def test_new_is_blank_clone(self):
        """Clone at new bypasses pretrained-state cloning."""
        forecaster = _make_pretrained_dummy()

        cloned = forecaster._clone_at("new")

        assert cloned is not forecaster
        assert cloned.strategy == forecaster.strategy
        assert cloned.state == "new"
        assert not hasattr(cloned, "global_mean_")
        assert not hasattr(cloned, "_pretrained_attrs")
        assert hasattr(forecaster, "global_mean_")

    def test_no_pretrain_capability_falls_back_to_clone(self):
        """Forecasters without pretrain capability use ordinary clone."""
        forecaster = NaiveForecaster()
        forecaster.fit(_make_series(n_timepoints=20), fh=[1, 2, 3])

        cloned = forecaster._clone_at("pretrained")

        assert not cloned.is_fitted

    def test_invalid_state_raises(self):
        """Only new and pretrained are valid target states."""
        with pytest.raises(ValueError, match="target state"):
            DummyGlobalForecaster()._clone_at("fitted")


def test_state_aware_tuner_sequence_preserves_pretraining():
    """_clone_at followed by _set_params_at keeps pretraining state."""
    forecaster = _make_pretrained_dummy()
    pretrain_mean = forecaster.global_mean_

    candidate = forecaster._clone_at("pretrained")
    candidate._set_params_at("pretrained", {"strategy": "last"})
    candidate.fit(_make_series(n_timepoints=20), fh=[1, 2, 3])

    assert candidate.is_fitted
    np.testing.assert_almost_equal(candidate.global_mean_, pretrain_mean)
