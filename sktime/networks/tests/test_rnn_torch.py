"""Tests for RNNNetworkTorch."""

import pytest

from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if torch is not available",
)
class TestRNNNetworkTorch:
    """Tests for RNNNetworkTorch."""

    def test_activation_module_instance(self):
        """Test that passing a torch.nn.Module instance as activation works.

        Regression test for https://github.com/sktime/sktime/issues/9627.
        """
        import torch

        from sktime.networks.rnn._rnn_torch import RNNNetworkTorch

        # This should not raise AttributeError
        network = RNNNetworkTorch(
            input_size=5,
            num_classes=2,
            activation=torch.nn.Sigmoid(),
        )
        assert network is not None

        # Verify forward pass works with module activation
        X = torch.randn(4, 3, 5)
        output = network(X)
        assert output.shape == (4, 2)

    def test_activation_string(self):
        """Test that passing a string activation still works after the fix."""
        from sktime.networks.rnn._rnn_torch import RNNNetworkTorch

        network = RNNNetworkTorch(
            input_size=5,
            num_classes=2,
            activation="sigmoid",
        )
        assert network is not None

    def test_activation_none(self):
        """Test that passing None as activation still works."""
        from sktime.networks.rnn._rnn_torch import RNNNetworkTorch

        network = RNNNetworkTorch(
            input_size=5,
            num_classes=2,
            activation=None,
        )
        assert network is not None
