"""Tests for RNN PyTorch network."""

import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none")
    or not run_test_module_changed("sktime.networks"),
    reason="skip test if required soft dependency not available",
)
def test_rnn_network_accepts_module_output_activation():
    """RNN should accept torch.nn.Module activation in output layer."""
    import torch

    from sktime.networks.rnn import RNNNetworkTorch

    network = RNNNetworkTorch(
        input_size=3,
        num_classes=2,
        activation=torch.nn.Sigmoid(),
    )

    X = torch.randn(4, 6, 3)
    out = network(X)

    assert tuple(out.shape) == (4, 2)
