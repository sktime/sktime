"""Pytorch Dataset for Transformer Classifier."""

import numpy as np

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class PytorchDataset(Dataset):
    """Dataset for use in sktime deep learning classifier based on pytorch."""

    def __init__(self, X, y):
        # X.shape = (batch_size, n_dims, n_timestamps)
        X = np.transpose(X, (0, 2, 1))
        # X.shape = (batch_size, n_timestamps, n_dims)

        self.X = X
        self.y = y

    def __len__(self):
        """Get length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Get item at index."""
        x = self.X[i]
        x = torch.tensor(x, dtype=torch.float)
        padding_masks = torch.ones(x.shape[:-1], dtype=torch.bool)

        inputs = {
            "X": x,
            "padding_masks": padding_masks,
        }

        # to make it reusable for predict
        if self.y is None:
            return inputs

        # return y during fit
        y = self.y[i]
        y = torch.tensor(y, dtype=torch.long)
        return inputs, y
