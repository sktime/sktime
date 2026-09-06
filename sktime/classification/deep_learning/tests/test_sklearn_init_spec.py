"""Tests for sklearn init specification conformance in deep learning classifiers.

Regression tests for #10208: parameters must be stored exactly as passed
to ``__init__``, so that ``get_params``, ``clone`` and ``set_params`` work.
"""

__author__ = ["Rythamo8055"]

import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.base import clone

from sktime.classification.deep_learning.gru import GRUFCNNClassifier
from sktime.classification.deep_learning.mcdcnn._mcdcnn_torch import (
    MCDCNNClassifierTorch,
)

torch_available = _check_soft_dependencies("torch", severity="none")


def test_grufcn_classifier_init_spec():
    """GRUFCNNClassifier stores optimizer_kwargs and layer params as passed."""
    clf = GRUFCNNClassifier(hidden_dim=8, gru_layers=1, optimizer_kwargs=None)
    params = clf.get_params()
    assert params["optimizer_kwargs"] is None
    assert params["conv_layers"] is None
    assert params["kernel_sizes"] is None

    # clone round-trip preserves parameters
    cloned = clone(clf)
    assert cloned.get_params() == params

    # explicit values are stored as-is
    clf2 = GRUFCNNClassifier(
        hidden_dim=8,
        gru_layers=1,
        conv_layers=[16, 32, 16],
        kernel_sizes=[3, 5, 3],
        optimizer_kwargs={"betas": (0.5, 0.99)},
    )
    params2 = clf2.get_params()
    assert params2["conv_layers"] == [16, 32, 16]
    assert params2["kernel_sizes"] == [3, 5, 3]
    assert params2["optimizer_kwargs"] == {"betas": (0.5, 0.99)}
    assert clone(clf2).get_params() == params2


@pytest.mark.skipif(not torch_available, reason="torch not available")
def test_mcdcnn_classifier_torch_init_spec():
    """MCDCNNClassifierTorch stores optim and optim_kwargs as passed."""
    clf = MCDCNNClassifierTorch(optim_kwargs=None)
    params = clf.get_params()
    assert params["optim"] is None
    assert params["optim_kwargs"] is None

    # clone round-trip preserves parameters
    cloned = clone(clf)
    assert cloned.get_params() == params

    # explicit values are stored as-is
    clf2 = MCDCNNClassifierTorch(optim="Adam", optim_kwargs={"lr": 0.05})
    params2 = clf2.get_params()
    assert params2["optim"] == "Adam"
    assert params2["optim_kwargs"] == {"lr": 0.05}
    assert clone(clf2).get_params() == params2


@pytest.mark.skipif(not torch_available, reason="torch required for fit test")
def test_grufcn_optimizer_defaults_preserved():
    """GRUFCNNClassifier applies default Adam betas when optimizer_kwargs is None."""
    X = np.random.randn(20, 1, 10).astype("float32")
    y = np.array([0, 1] * 10)

    clf = GRUFCNNClassifier(
        hidden_dim=8, gru_layers=1, num_epochs=1, batch_size=8, lr=0.01
    )
    clf.fit(X, y)
    assert clf._optimizer.param_groups[0]["betas"] == (0.9, 0.999)
