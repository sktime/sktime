"""Tests for state-aware clone and set_params functions on arbitrary objects."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
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


def _make_pretrained_dummy(strategy="mean"):
    """Return a pretrained DummyGlobalForecaster."""
    y_panel = _make_hierarchical(
        hierarchy_levels=(3,), min_timepoints=10, max_timepoints=10, n_columns=1
    )
    return DummyGlobalForecaster(strategy=strategy).pretrain(y_panel)


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


def test_pretrained_cloner_subclass_handled_like_pretrained_cloner():
    """Subclasses of _PretrainedCloner are not duplicated, and skipped for "new"."""

    class _CustomCloner(_PretrainedCloner):
        pass

    class _ParentWithSubclass(BaseObject):
        @classmethod
        def _get_clone_plugins(cls):
            return [_CustomCloner, _KeepFoo]

    class _Child(_PretrainMixin, _ParentWithSubclass):
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


def test_clone_at_shares_pretrained_attributes_reference():
    """Clones share references to pretrained attributes to prevent memory explosion."""
    forecaster = _make_pretrained_dummy(strategy="mean_by_index")

    cloned = _clone_at(forecaster, "pretrained")
    assert cloned is not forecaster
    assert cloned.state == "pretrained"
    assert getattr(cloned, "_has_shared_pretrained_attrs", False) is True
    assert getattr(forecaster, "_has_shared_pretrained_attrs", False) is True
    # Reference sharing check for non-scalar pretrained attribute
    assert cloned.mean_by_index_ is forecaster.mean_by_index_


def test_pretrain_mutation_detaches_shared_pretrained_attributes():
    """Mutating operations detach shared pretrained attributes via copy-on-write."""
    forecaster = _make_pretrained_dummy(strategy="mean_by_index")
    original_mean_by_index = forecaster.mean_by_index_

    cloned = _clone_at(forecaster, "pretrained")
    assert cloned.mean_by_index_ is original_mean_by_index

    # Pretrain clone on different panel data
    y_panel_new = _make_hierarchical(
        hierarchy_levels=(2,), min_timepoints=5, max_timepoints=5, n_columns=1
    )
    cloned.pretrain(y_panel_new)

    assert cloned._has_shared_pretrained_attrs is False
    assert cloned.mean_by_index_ is not original_mean_by_index
    assert forecaster.mean_by_index_ is original_mean_by_index


def test_grid_search_multiclone_memory_footprint():
    """Simulated grid-search clone sweep verifies all clones share memory."""
    forecaster = _make_pretrained_dummy(strategy="mean_by_index")
    clones = [_clone_at(forecaster, "pretrained") for _ in range(10)]

    for c in clones:
        assert c is not forecaster
        assert c.mean_by_index_ is forecaster.mean_by_index_
        assert c._has_shared_pretrained_attrs is True


def test_clone_at_pretrained_fit_isolation():
    """Verify clones fitted on different data are isolated."""
    f = _make_pretrained_dummy(strategy="last")

    c1 = _clone_at(f, "pretrained")
    c2 = _clone_at(f, "pretrained")

    y1 = _make_series(n_timepoints=15)
    y2 = _make_series(n_timepoints=15) * 2.0

    c1.fit(y1, fh=[1])
    c2.fit(y2, fh=[1])

    # Pretrained base forecaster remains unfitted and untouched
    assert not f.is_fitted
    assert f.state == "pretrained"
    assert not hasattr(f, "last_value_")

    # Clones are fitted and independent
    assert c1.is_fitted and c2.is_fitted
    assert c1.state == "fitted" and c2.state == "fitted"
    assert c1.last_value_ is not c2.last_value_
    assert c1.last_value_ != c2.last_value_

    # Forecasts are independent
    pred1 = c1.predict()
    pred2 = c2.predict()
    assert not (pred1 == pred2).all()

    # Pretrained attributes remain shared and consistent
    assert c1.global_mean_ == f.global_mean_ == c2.global_mean_


def test_clone_at_pretrained_fit_inplace_mutation_isolation():
    """Verify in-place mutation on clone after fit does not mutate base or sibling."""
    forecaster = _make_pretrained_dummy(strategy="mean_by_index")
    original_series = forecaster.mean_by_index_.copy()

    c1 = _clone_at(forecaster, "pretrained")
    c2 = _clone_at(forecaster, "pretrained")

    # Before fit, reference is shared
    assert c1.mean_by_index_ is forecaster.mean_by_index_
    assert c2.mean_by_index_ is forecaster.mean_by_index_

    y1 = _make_series(n_timepoints=15)
    c1.fit(y1, fh=[1])

    # Calling fit detaches c1 via copy-on-write
    assert c1._has_shared_pretrained_attrs is False
    assert c1.mean_by_index_ is not forecaster.mean_by_index_

    # Simulate in-place mutation on c1's pretrained attribute
    c1.mean_by_index_.iloc[0, 0] = 99999.0

    # Base estimator and sibling c2 are completely protected
    pd.testing.assert_frame_equal(forecaster.mean_by_index_, original_series)
    assert forecaster.mean_by_index_.iloc[0, 0] != 99999.0
    assert c2.mean_by_index_.iloc[0, 0] != 99999.0

    # Refit on c1 restores clean pretrained state via _reset_at("pretrained")
    y2 = _make_series(n_timepoints=15) * 1.5
    c1.fit(y2, fh=[1])
    assert c1.is_fitted
