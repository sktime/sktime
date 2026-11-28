# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for singleton and multiton decorators."""

import pytest

from sktime.forecasting.dummy_global import DummyGlobalForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.singleton import (
    _multiton,
    _MultitonPretrainMixin,
    _singleton,
)


@_singleton
class _TestSingleton:
    def __init__(self, a=0):
        self.a = a

    def set_b(self, b):
        self.b = b


@_multiton
class _TestMultiton:
    def __init__(self, key, a=0):
        self.key = key
        self.a = a

    def set_b(self, b):
        self.b = b


@_multiton
class _TestPretrainCache:
    def __init__(self, key):
        self.key = key
        self.model_ = None

    def load(self):
        if self.model_ is None:
            self.model_ = {"weights": [1]}
        return self.model_


class _TestMultitonForecaster(_MultitonPretrainMixin, DummyGlobalForecaster):
    _tags = {
        **DummyGlobalForecaster._tags,
        "pretrain:fitted_params": [
            "global_mean_",
            "global_std_",
            "n_pretrain_instances_",
            "n_pretrain_timepoints_",
            "model_",
            "_multiton_key",
        ],
    }

    def __init__(self):
        self._multiton_key = None
        super().__init__()

    def _get_multiton_cache_class(self):
        return _TestPretrainCache

    def _get_loading_param_key(self):
        return "test-loading-key"

    def _get_multiton_cache_kwargs(self):
        return {}

    def _load_model(self):
        if hasattr(self, "model_") and self.model_ is not None:
            return self.model_
        self.model_ = _TestPretrainCache(
            key=self._get_unique_key(),
        ).load()
        return self.model_

    def _pretrain(self, y, X=None, fh=None):
        super()._pretrain(y=y, X=X, fh=fh)
        self.model_["weights"][0] += 1
        return self

    def _pretrain_update(self, y, X=None, fh=None):
        return self._pretrain(y=y, X=X, fh=fh)


@pytest.mark.skipif(
    not run_test_for_class(_singleton),
    reason="run test incrementally (if requested)",
)
def test_singleton():
    """Test singleton behaviour."""
    instance1 = _TestSingleton(42)
    instance1.set_b(43)
    instance2 = _TestSingleton()

    assert instance1 is instance2
    assert instance1.a == 42
    assert instance2.a == 42

    assert instance1.b == 43
    assert instance2.b == 43

    instance2.set_b(41)
    assert instance1.b == 41


@pytest.mark.skipif(
    not run_test_for_class(_multiton),
    reason="run test incrementally (if requested)",
)
def test_multiton():
    """Test multiton behaviour."""
    instance1 = _TestMultiton("key1", 42)
    instance1.set_b(43)
    instance2 = _TestMultiton("key2")
    instance2.set_b(44)

    instance1b = _TestMultiton("key1", 77)
    instance2b = _TestMultiton("key2")

    assert instance1 is not instance2
    assert instance1 is instance1b
    assert instance2 is instance2b

    assert instance1.a == 42
    assert instance2.a == 0
    assert instance1b.b == 43
    assert instance2.b == 44


def test_multiton_pretrain_detach_and_state():
    """Pretraining detaches fake models and preserves pretrained state."""
    y = _make_hierarchical(
        hierarchy_levels=(4,),
        min_timepoints=20,
        max_timepoints=20,
        n_columns=1,
    )

    base_entry = _TestPretrainCache("test-loading-key")
    base_entry.load()
    base_model = base_entry.model_

    pretrained = _TestMultitonForecaster()
    pretrained._detach_pretrained_multiton()
    assert pretrained._multiton_key is not None
    assert pretrained.model_ is not base_model
    assert pretrained.model_ == base_model
    pretrained = pretrained.pretrain(y)

    pretrained.model_["weights"][0] = 99
    assert base_entry.model_ == {"weights": [1]}

    zero_shot = _TestMultitonForecaster()
    assert zero_shot._load_model() == {"weights": [1]}
    assert zero_shot.model_ is base_entry.model_

    pretrained.model_["weights"][0] = 2
    clone = pretrained._clone_at("pretrained")
    assert clone.model_ is pretrained.model_
    assert clone._multiton_key is pretrained._multiton_key

    parent_model = pretrained.model_
    parent_weight = parent_model["weights"][0]
    parent_key = clone._multiton_key
    clone.pretrain(y)
    assert pretrained.model_ is parent_model
    assert pretrained.model_["weights"][0] == parent_weight
    assert clone._multiton_key != parent_key
    assert clone._multiton_key is not None

    accumulated = _TestMultitonForecaster()
    accumulated.pretrain(y)
    first_key = accumulated._multiton_key
    accumulated.pretrain(y)
    assert accumulated._multiton_key != first_key
    assert hasattr(accumulated, "global_mean_")
    assert hasattr(accumulated, "n_pretrain_instances_")

    reset_clone = accumulated._reset_at("pretrained")
    assert reset_clone._multiton_key == accumulated._multiton_key
    assert reset_clone.model_ is accumulated.model_

    plain_reset = accumulated.reset()
    assert plain_reset._multiton_key is None
    assert not hasattr(plain_reset, "model_")
    assert "_multiton_key" not in plain_reset.get_params(deep=True)


def test_multiton_pretrain_mixin_mro():
    """Affected forecasters expose shared detach and key implementations."""
    forecasters = []
    for module_name, class_name in [
        ("sktime.forecasting.sundial", "SundialForecaster"),
        ("sktime.forecasting.timemoe", "TimeMoEForecaster"),
        ("sktime.forecasting.timesfm2", "TimesFM2Forecaster"),
        ("sktime.forecasting.ttm", "TinyTimeMixerForecaster"),
        ("sktime.forecasting.toto", "TotoForecaster"),
    ]:
        try:
            module = __import__(module_name, fromlist=[class_name])
            forecasters.append(getattr(module, class_name))
        except (ImportError, ModuleNotFoundError):
            continue

    for forecaster in forecasters:
        assert getattr(forecaster, "_detach_pretrained_multiton") is getattr(
            _MultitonPretrainMixin, "_detach_pretrained_multiton"
        )
        assert getattr(forecaster, "_get_unique_key") is getattr(
            _MultitonPretrainMixin, "_get_unique_key"
        )
