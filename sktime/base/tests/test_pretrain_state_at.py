"""Tests for state-aware clone and set_params functions on arbitrary objects."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pytest
from skbase.base import BaseObject
from skbase.base._clone_plugins import _CloneSkbase
from sklearn.linear_model import LinearRegression

from sktime.base._pretrain import _clone_at, _set_params_at
from sktime.base._pretrain._clone_plugin import _PretrainedCloner
from sktime.base._pretrain._mixin import _PretrainMixin
from sktime.forecasting.dummy_global import DummyGlobalForecaster
from sktime.forecasting.naive import NaiveForecaster
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


class _KeepFoo(_CloneSkbase):
    """Custom clone plugin that carries over the attribute ``foo``."""

    def _check(self, obj):
        return isinstance(obj, (_PluginHolder, _NonCapableHolder))

    def _clone(self, obj):
        clone = super()._clone(obj)
        clone.foo = obj.foo
        return clone


class _PluginHolder(DummyGlobalForecaster):
    """Pretrain capable forecaster with a component and a custom clone plugin."""

    def __init__(self, inner=None, strategy="mean"):
        self.inner = inner
        super().__init__(strategy=strategy)

    @classmethod
    def _get_clone_plugins(cls):
        return [_KeepFoo, *super()._get_clone_plugins()]


class _NonCapableHolder(NaiveForecaster):
    """Composite without pretrain capability, holding a component."""

    def __init__(self, inner=None):
        self.inner = inner
        super().__init__()

    @classmethod
    def _get_clone_plugins(cls):
        return [_KeepFoo, *super()._get_clone_plugins()]


def test_get_clone_plugins_keeps_parent_plugins():
    """_PretrainMixin adds _PretrainedCloner, and keeps plugins of parents."""

    class _ParentWithPlugin(BaseObject):
        @classmethod
        def _get_clone_plugins(cls):
            return [_KeepFoo]

    class _ChildWithPlugin(_PretrainMixin, _ParentWithPlugin):
        pass

    class _ChildWithoutPlugin(_PretrainMixin, BaseObject):
        pass

    assert _ChildWithPlugin._get_clone_plugins() == [_PretrainedCloner, _KeepFoo]
    assert _ChildWithoutPlugin._get_clone_plugins() == [_PretrainedCloner]


def test_get_clone_plugins_no_duplicate_pretrained_cloner():
    """_PretrainedCloner is not duplicated if a parent already returns it."""

    class _ParentWithCloner(BaseObject):
        @classmethod
        def _get_clone_plugins(cls):
            return [_PretrainedCloner, _KeepFoo]

    class _Child(_PretrainMixin, _ParentWithCloner):
        pass

    assert _Child._get_clone_plugins() == [_PretrainedCloner, _KeepFoo]


def test_clone_at_new_keeps_other_clone_plugins():
    """_clone_at("new") only skips _PretrainedCloner, other plugins still apply."""
    forecaster = _PluginHolder(inner=_make_pretrained_dummy())
    forecaster.foo = "keep me"

    assert forecaster.clone().foo == "keep me"

    blank = _clone_at(forecaster, "new")
    assert blank.foo == "keep me"
    assert blank.state == "new"


def test_clone_at_new_applies_state_to_components():
    """_clone_at("new") clones components without pretrained state, clone keeps it."""
    forecaster = _PluginHolder(inner=_make_pretrained_dummy())
    forecaster.foo = "keep me"

    assert forecaster.clone().inner.state == "pretrained"

    blank = forecaster._clone_at("new")
    assert blank.inner is not forecaster.inner
    assert blank.inner.state == "new"
    assert not hasattr(blank.inner, "global_mean_")
    assert forecaster.inner.state == "pretrained"


def test_clone_at_pretrained_keeps_components_state():
    """_clone_at("pretrained") is equivalent to clone, including components."""
    forecaster = _PluginHolder(inner=_make_pretrained_dummy())
    forecaster.foo = "keep me"

    cloned = forecaster._clone_at("pretrained")
    assert cloned.foo == "keep me"
    assert cloned.inner.state == "pretrained"
    np.testing.assert_almost_equal(
        cloned.inner.global_mean_, forecaster.inner.global_mean_
    )


def test_clone_at_new_passes_state_through_non_capable_composite():
    """Composites without pretrain capability still pass "new" on to components."""
    composite = _NonCapableHolder(inner=_make_pretrained_dummy())
    composite.foo = "keep me"
    assert not composite._has_pretrain_capability()

    assert composite.clone().inner.state == "pretrained"

    blank = composite._clone_at("new")
    assert blank.foo == "keep me"
    assert blank.inner.state == "new"
    assert not hasattr(blank.inner, "global_mean_")
