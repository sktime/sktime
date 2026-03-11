"""Tests for activation validation/behavior in BaseDeepRegressorTorch."""

import numpy as np
import pytest

from sktime.networks.utils import instantiate_activation
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch
from sktime.sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(BaseDeepRegressorTorch, severity="none"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)

import torch


class _DummyRegressorNet(torch.nn.Module):
    def __init__(self, in_features, activation=None):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)
        self.activation = activation

    def forward(self, X):
        x = torch.flatten(X, start_dim=1)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x.squeeze(-1)


class DummyRegressorTorch(BaseDeepRegressorTorch):
    """Minimal regressor to isolate BaseDeepRegressorTorch behavior."""

    _validate_activation_vars = ("activation", "foo_activation")
    _supported_activation = ("sigmoid", "relu", "tanh")
    _supported_foo_activation = ("relu", "tanh")

    def __init__(
        self,
        activation=None,
        foo_activation="relu",
        random_state=0,
    ):
        self.activation = activation
        self.foo_activation = foo_activation
        super().__init__(
            num_epochs=1,
            batch_size=2,
            criterion="mseloss",
            optimizer="adam",
            lr=0.01,
            verbose=False,
            random_state=random_state,
        )

    def _build_network(self, X):
        in_features = int(X.shape[1] * X.shape[2])
        activation = instantiate_activation(self.activation)
        return _DummyRegressorNet(in_features=in_features, activation=activation)


def _make_regression_data():
    rng = np.random.RandomState(11)
    X = rng.randn(12, 2, 5).astype(np.float32)
    y = rng.randn(12).astype(np.float32)
    X_test = rng.randn(4, 2, 5).astype(np.float32)
    return X, y, X_test


def test_regressor_rejects_unsupported_activation_string():
    with pytest.raises(ValueError, match="`activation` must be one of"):
        DummyRegressorTorch(activation="softmax")


def test_regressor_rejects_unsupported_custom_activation_var():
    with pytest.raises(ValueError, match="`foo_activation` must be one of"):
        DummyRegressorTorch(foo_activation="gelu")


def test_regressor_allows_torch_module_instances_for_activation_vars():
    est = DummyRegressorTorch(
        activation=torch.nn.Sigmoid(),
        foo_activation=torch.nn.ReLU(),
    )
    X, y, X_test = _make_regression_data()
    est.fit(X, y)
    y_pred = est.predict(X_test)
    assert y_pred.shape == (len(X_test),)


def test_regressor_allows_none_activation_and_fit_predict_runs():
    est = DummyRegressorTorch(activation=None, foo_activation=None)
    X, y, X_test = _make_regression_data()
    est.fit(X, y)
    y_pred = est.predict(X_test)
    assert y_pred.shape == (len(X_test),)
    assert np.all(np.isfinite(y_pred))


def test_regressor_string_activation_fit_predict_runs():
    est = DummyRegressorTorch(activation="sigmoid", foo_activation="tanh")
    X, y, X_test = _make_regression_data()
    est.fit(X, y)
    y_pred = est.predict(X_test)
    assert y_pred.shape == (len(X_test),)
    assert np.all((y_pred >= 0.0) & (y_pred <= 1.0))
