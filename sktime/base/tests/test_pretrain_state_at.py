"""Tests for state-aware clone and set_params functions on arbitrary objects."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from sktime.base._pretrain import _clone_at, _set_params_at
from sktime.forecasting.dummy_global import DummyGlobalForecaster
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series


def _make_pretrained_dummy():
    """Return a pretrained DummyGlobalForecaster."""
    y_panel = _make_hierarchical(
        hierarchy_levels=(3,), min_timepoints=10, max_timepoints=10, n_columns=1
    )
    return DummyGlobalForecaster().pretrain(y_panel)


def test_clone_at_delegates_to_estimator():
    """Objects with _clone_at are cloned state-aware."""
    forecaster = _make_pretrained_dummy()

    cloned = _clone_at(forecaster, "pretrained")
    assert cloned is not forecaster
    assert cloned.state == "pretrained"
    np.testing.assert_almost_equal(cloned.global_mean_, forecaster.global_mean_)

    blank = _clone_at(forecaster, "new")
    assert blank.state == "new"
    assert not hasattr(blank, "global_mean_")


def test_clone_at_falls_back_to_clone():
    """Objects without _clone_at get an ordinary blank clone."""
    transformer = ExponentTransformer(power=3)
    transformer.fit(_make_series(n_timepoints=20))
    assert not hasattr(transformer, "_clone_at")

    cloned = _clone_at(transformer, "pretrained")
    assert cloned is not transformer
    assert cloned.power == 3
    assert not cloned.is_fitted

    regressor = LinearRegression(fit_intercept=False)
    regressor.fit(np.ones((3, 1)), np.ones(3))

    cloned = _clone_at(regressor, "pretrained")
    assert cloned is not regressor
    assert not cloned.fit_intercept
    assert not hasattr(cloned, "coef_")


def test_set_params_at_delegates_to_estimator():
    """Objects with _set_params_at keep pretrained state on set_params."""
    forecaster = _make_pretrained_dummy()
    pretrain_mean = forecaster.global_mean_

    result = _set_params_at(forecaster, "pretrained", {"strategy": "last"})

    assert result is forecaster
    assert forecaster.strategy == "last"
    assert forecaster.state == "pretrained"
    np.testing.assert_almost_equal(forecaster.global_mean_, pretrain_mean)


def test_set_params_at_falls_back_to_set_params():
    """Objects without _set_params_at get an ordinary set_params call."""
    transformer = ExponentTransformer(power=3)
    _set_params_at(transformer, "pretrained", {"power": 2})
    assert transformer.power == 2

    regressor = LinearRegression()
    _set_params_at(regressor, "pretrained", {"fit_intercept": False})
    assert not regressor.fit_intercept


@pytest.mark.parametrize(
    "obj", [DummyGlobalForecaster(), ExponentTransformer(), LinearRegression()]
)
def test_invalid_state_raises(obj):
    """Invalid target states raise, also for objects without the methods."""
    with pytest.raises(ValueError, match="target state"):
        _clone_at(obj, "fitted")
    with pytest.raises(ValueError, match="target state"):
        _set_params_at(obj, "fitted", {})
