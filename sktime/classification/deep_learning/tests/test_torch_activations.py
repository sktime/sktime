"""Tests for activation validation/behavior in BaseDeepClassifierPytorch."""

import numpy as np
import pytest

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.utils import instantiate_activation
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(BaseDeepClassifierPytorch, severity="none"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)

from torch import cat, flatten, from_numpy, no_grad
from torch.nn import (
    Linear,
    Module,
    ReLU,
    Sigmoid,
    functional,
)


class _DummyClassifierNet(Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.linear = Linear(in_features, out_features)
        self.activation = activation

    def forward(self, X):
        x = flatten(X, start_dim=1)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DummyClassifierTorch(BaseDeepClassifierPytorch):
    """Minimal classifier to isolate BaseDeepClassifierPytorch behavior."""

    _validate_activation_vars = ("activation", "foo_activation")
    _supported_activation = ("sigmoid", "softmax", "logsoftmax", "logsigmoid")
    _supported_foo_activation = ("relu", "tanh")

    def __init__(
        self,
        activation=None,
        foo_activation="relu",
        criterion="crossentropyloss",
        random_state=0,
    ):
        self.foo_activation = foo_activation
        super().__init__(
            num_epochs=1,
            batch_size=2,
            activation=activation,
            criterion=criterion,
            optimizer="adam",
            lr=0.01,
            verbose=False,
            random_state=random_state,
        )

    def _build_network(self, X, y):
        in_features = int(X.shape[1] * X.shape[2])
        n_classes = len(np.unique(y))
        activation = instantiate_activation(self._validated_activation)
        return _DummyClassifierNet(
            in_features=in_features,
            out_features=n_classes,
            activation=activation,
        )


def _make_classification_data():
    rng = np.random.RandomState(7)
    X = rng.randn(12, 2, 5).astype(np.float32)
    y = np.array([0, 1] * 6)
    X_test = rng.randn(4, 2, 5).astype(np.float32)
    return X, y, X_test


def _network_outputs(est, X):
    est.network.eval()
    outputs = []
    with no_grad():
        for inputs in est._build_dataloader(X):
            outputs.append(est.network(**inputs).detach())
    return cat(outputs, dim=0).numpy()


def test_classifier_rejects_unsupported_activation_string():
    with pytest.raises(ValueError, match="`activation` must be one of"):
        DummyClassifierTorch(activation="relu", criterion="mseloss")


def test_classifier_rejects_unsupported_custom_activation_var():
    with pytest.raises(ValueError, match="`foo_activation` must be one of"):
        DummyClassifierTorch(foo_activation="gelu")


def test_classifier_allows_torch_module_instances_for_activation_vars():
    sigmoid = Sigmoid()
    relu = ReLU()
    est = DummyClassifierTorch(
        activation=sigmoid,
        foo_activation=relu,
        criterion="mseloss",
    )
    assert est.activation is sigmoid
    assert est.foo_activation is relu


def test_classifier_activation_none_applies_softmax_in_predict_proba():
    est = DummyClassifierTorch(activation=None, criterion="crossentropyloss")
    X, y, X_test = _make_classification_data()
    est.fit(X, y)

    proba = est.predict_proba(X_test)
    raw = _network_outputs(est, X_test)

    expected = functional.softmax(from_numpy(raw), dim=-1).numpy()
    np.testing.assert_allclose(proba, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_test)), atol=1e-6)


def test_classifier_sigmoid_with_mse_keeps_sigmoid_and_skips_softmax():
    est = DummyClassifierTorch(activation="sigmoid", criterion="mseloss")
    assert est._validated_activation == "sigmoid"
    assert str(est._validated_criterion).lower() == "mseloss"
