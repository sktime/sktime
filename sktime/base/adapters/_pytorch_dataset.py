"""Dataset for sktime deep learning estimators based on pytorch."""

__all__ = ["PytorchDataset"]
__authors__ = ["geetu040", "RecreationalMath", "srupat"]

import numpy as np

from sktime.utils.dependencies import _safe_import

Dataset = _safe_import("torch.utils.data.Dataset")


class PytorchDataset(Dataset):
    """Dataset for use in sktime deep learning estimators based on pytorch."""

    def __init__(self, X, y=None, y_dtype="float"):
        # X.shape = (batch_size, n_dims, n_timestamps)
        X = np.transpose(X, (0, 2, 1))
        # X.shape = (batch_size, n_timestamps, n_dims)

        self.X = X
        self.y = y
        self.y_dtype = y_dtype

    def __len__(self):
        """Get length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Get item at index."""
        torchTensor = _safe_import("torch.tensor")
        x = torchTensor(self.X[i], dtype=_safe_import("torch.float"))
        inputs = {"X": x}
        # to make it reusable for predict
        if self.y is None:
            return inputs

        # return y during fit
        y = torchTensor(self.y[i], dtype=_safe_import(f"torch.{self.y_dtype}"))
        return inputs, y
