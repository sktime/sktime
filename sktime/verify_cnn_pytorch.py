import torch
import numpy as np
from sktime.classification.deep_learning.cnn_pytorch import CNNClassifier

def verify_cnn():
    print("1. Initializing CNNClassifier...")
    clf = CNNClassifier(n_epochs=2, batch_size=2, verbose=True)
    
    print("2. Generating dummy data...")
    # sktime format: (n_samples, n_channels, n_timepoints) or (n, d, m)
    X = np.random.randn(10, 1, 50).astype(np.float32)
    # y: (n_samples)
    y = np.random.randint(0, 2, size=(10,))
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    print("3. Fitting model...")
    try:
        clf.fit(X, y)
        print("   Allowed fit to complete.")
    except Exception as e:
        print(f"   !!! Fit failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("4. Predicting...")
    try:
        y_pred = clf.predict(X)
        print(f"   Prediction shape: {y_pred.shape}")
        print(f"   Predictions: {y_pred}")
    except Exception as e:
        print(f"   !!! Predict failed: {e}")
        return

    print("5. Success! The PyTorch CNN runs end-to-end.")

if __name__ == "__main__":
    try:
        import sktime
        print(f"sktime imported from: {sktime.__file__}")
        verify_cnn()
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Make sure you are running this from the repo root.")
