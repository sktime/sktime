# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for shared pretrain state transitions."""

import pytest

from sktime.base import BaseEstimator
from sktime.base._pretrain._mixin import _PretrainMixin
from sktime.tests.test_switch import run_test_module_changed

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.base._pretrain"),
    reason="run only if pretrain module has changed",
)


class _IncrementalEstimator(_PretrainMixin, BaseEstimator):
    """Estimator whose pretrain update requires existing learned state."""

    _tags = {"capability:pretrain": True}

    def __init__(self):
        self._state = "new"
        self._batches = []
        super().__init__()

    def pretrain(self, value):
        return self._pretrain_and_track(value=value)

    def _pretrain(self, value):
        self.total_ = value
        self._batches.append(value)

    def _pretrain_update(self, value):
        self.total_ += value
        self._batches.append(value)
        self.updated_ = True

    def fit(self):
        self.fitted_value_ = 10
        self._state = "fitted"
        return self


@pytest.mark.parametrize("pretrained", [False, True])
def test_pretrain_after_fit(pretrained):
    """Pretraining clears fitted state and initializes or updates learned state."""
    estimator = _IncrementalEstimator()
    if pretrained:
        estimator.pretrain(1)
    estimator.fit()

    result = estimator.pretrain(2)

    assert result is estimator
    assert estimator.state == "pretrained"
    assert not estimator.is_fitted
    assert not hasattr(estimator, "fitted_value_")
    assert estimator.total_ == (3 if pretrained else 2)
    expected_params = {"total_": 3, "updated_": True} if pretrained else {"total_": 2}
    assert estimator.get_pretrained_params() == expected_params


def test_pretrain_update_preserves_existing_state():
    """Repeated pretraining preserves state needed by incremental updates."""
    estimator = _IncrementalEstimator().pretrain(1)

    estimator.pretrain(2)

    assert estimator.state == "pretrained"
    assert estimator.total_ == 3
    assert estimator._batches == [1, 2]
    assert estimator.get_pretrained_params() == {"total_": 3, "updated_": True}
