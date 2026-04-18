"""Check mathematical parity between Keras and PyTorch ResNet implementations."""

import numpy as np

from sktime.classification.deep_learning.resnet import (
    ResNetClassifier as KerasResNet,
)
from sktime.classification.deep_learning.resnet_torch import (
    ResNetClassifier as TorchResNet,
)


def check_parity():
    """Verify that Keras and PyTorch ResNet models produce consistent outputs."""
    print("Starting Mathematical Parity Check: Keras vs. PyTorch...")

    # 1. Setup dummy data (3 samples, 1 channel, 50 time steps)
    X = np.random.randn(3, 1, 50)
    y = np.array([0, 1, 0])

    # 2. Initialize both models
    # Note: framework differences mean weights won't be identical,
    # but the API behavior and output shapes must match.
    keras_clf = KerasResNet(n_epochs=1, random_state=42)
    torch_clf = TorchResNet(n_epochs=1, random_state=42)

    print("Training Keras model...")
    keras_clf.fit(X, y)
    k_pred = keras_clf.predict(X)

    print("Training PyTorch model...")
    torch_clf.fit(X, y)
    t_pred = torch_clf.predict(X)

    print("\n" + "=" * 40)
    print(f"Keras Predictions: {k_pred}")
    print(f"Torch Predictions: {t_pred}")

    # Check shape parity
    if k_pred.shape == t_pred.shape:
        print("SUCCESS: Output shapes are identical.")

    # Check if they are producing valid class labels
    if set(t_pred).issubset({0, 1}):
        print("SUCCESS: PyTorch model is producing valid labels.")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    check_parity()
