"""Tests for BaseDeepClassifierPytorch activation warning behavior."""

__author__ = ["achieveordie"]

import pytest

torch = pytest.importorskip("torch")

import warnings
import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.classification.deep_learning.mlp import MLPClassifierTorch


class TestClassifier(BaseDeepClassifierPytorch):
    """Simple test classifier for testing base class behavior."""

    def _build_network(self, X, y):
        """Simple network build for testing."""
        from sktime.utils.dependencies import _safe_import
        
        nnModule = _safe_import("torch.nn.Module")
        nnLinear = _safe_import("torch.nn.Linear")
        
        class SimpleNet(nnModule):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc = nnLinear(input_size, num_classes)
            
            def forward(self, x):
                return self.fc(x)
        
        num_classes = len(np.unique(y))
        input_size = X.shape[1] * X.shape[2]
        
        return SimpleNet(input_size, num_classes)


class TestActivationWarning:
    """Test activation parameter warning behavior."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        # Create simple test data
        self.X_train = np.random.random((10, 3, 20))  # 10 samples, 3 dims, 20 timesteps
        self.y_train = np.random.randint(0, 2, 10)   # Binary classification

    def test_bce_loss_sigmoid_override_with_warning(self):
        """Test BCELoss + sigmoid emits UserWarning and overrides to crossentropyloss + None."""
        with pytest.warns(UserWarning, match="overridden"):
            classifier = TestClassifier(
                criterion="BCELoss",
                activation="sigmoid",
                num_epochs=1,
                verbose=False
            )
        
        # After override, criterion should be crossentropyloss and activation should be None
        assert classifier._validated_criterion == "crossentropyloss"
        assert classifier._validated_activation is None

    def test_nll_loss_logsoftmax_override_with_warning(self):
        """Test NLLLoss + logsoftmax emits UserWarning and overrides to crossentropyloss + None."""
        with pytest.warns(UserWarning, match="overridden"):
            classifier = TestClassifier(
                criterion="NLLLoss",
                activation="logsoftmax",
                num_epochs=1,
                verbose=False
            )
        
        # After override, criterion should be crossentropyloss and activation should be None
        assert classifier._validated_criterion == "crossentropyloss"
        assert classifier._validated_activation is None

    def test_bcewithlogits_loss_none_override_no_warning(self):
        """Test BCEWithLogitsLoss + None overrides to crossentropyloss + None with NO warning."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            classifier = TestClassifier(
                criterion="BCEWithLogitsLoss",
                activation=None,
                num_epochs=1,
                verbose=False
            )
        
        # Should emit no warnings since activation was None to begin with
        assert len(record) == 0
        
        # After override, criterion should be crossentropyloss and activation should be None
        assert classifier._validated_criterion == "crossentropyloss"
        assert classifier._validated_activation is None

    def test_mlp_classifier_torch_warning_behavior(self):
        """Test warning behavior with actual MLPClassifierTorch implementation."""
        with pytest.warns(UserWarning, match="overridden"):
            classifier = MLPClassifierTorch(
                criterion="BCELoss",
                activation="sigmoid",
                num_epochs=1,
                verbose=False,
                hidden_dim=5,
                n_layers=1
            )
        
        # After override, criterion should be crossentropyloss and activation should be None
        assert classifier._validated_criterion == "crossentropyloss"
        assert classifier._validated_activation is None
