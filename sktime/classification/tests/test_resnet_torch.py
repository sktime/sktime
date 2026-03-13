import numpy as np
import torch
import pytest
from sktime.classification.deep_learning.resnet_torch import ResNetClassifier

def test_resnet_torch_integration():
    """Test the full lifecycle of the ResNetClassifier."""
    print("Starting ResNet PyTorch Integration Test...")
    
    # 1. Create dummy data (3 samples, 1 channel, 50 time-steps)
    X = np.random.randn(3, 1, 50)
    y = np.array([0, 1, 0])
    
    # 2. Initialize the classifier with minimal epochs for speed
    clf = ResNetClassifier(n_epochs=2, batch_size=2)
    
    try:
        # 3. Test Fit
        print("Testing .fit()...")
        clf.fit(X, y)
        
        # 4. Test Predict
        print("Testing .predict()...")
        y_pred = clf.predict(X)
        
        # 5. Assertions (The technical check)
        assert isinstance(y_pred, np.ndarray), "Output should be a numpy array"
        assert len(y_pred) == 3, "Output length should match input samples"
        
        print("✅ ALL TESTS PASSED: ResNet PyTorch is functional!")
        
    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")

if __name__ == "__main__":
    test_resnet_torch_integration()