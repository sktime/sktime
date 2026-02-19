import numpy as np
import pytest
from sktime.classification.deep_learning.cnn_pytorch import CNNClassifierTorch
from sktime.utils.dependencies import _check_soft_dependencies

@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_cnn_classifier_fit_predict():
    """Test CNNClassifierTorch fit and predict on dummy data."""
    # Create dummy data
    X = np.random.randn(10, 1, 50).astype(np.float32)
    y = np.random.randint(0, 2, size=(10,))
    
    # Initialize model
    clf = CNNClassifierTorch(n_epochs=2, batch_size=2)
    
    # Fit
    clf.fit(X, y)
    
    # Predict
    y_pred = clf.predict(X)
    
    # Check shape
    assert y_pred.shape == (10,)
