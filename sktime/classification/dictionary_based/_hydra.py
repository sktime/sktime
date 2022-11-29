# -*- coding: utf-8 -*-
"""HYDRA classifier.

HYDRA: Competing convolutional kernels for fast and accurate time series classification
By Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb
https://arxiv.org/abs/2203.13652
"""

__author__ = ["patrickzib", "Arik Ermshaus"]
__all__ = ["HYDRA"]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifierCV

from sktime.classification.base import BaseClassifier


class HYDRA(BaseClassifier):
    """Sktime-HYDRA classifier-adaptor."""

    _tags = {
        "capability:multithreading": False,
        "classifier_type": "dictionary",
    }

    def __init__(self, k=8, g=64):

        self.k = k
        self.g = g

        super(HYDRA, self).__init__()

    def _fit(self, X, y):
        self.transform = HydraInternal(X.shape[-1])
        X_training_transform = self.transform(torch.tensor(X).float())

        self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        self.clf.fit(X_training_transform, y)

    def _predict(self, X) -> np.ndarray:
        X_test_transform = self.transform(torch.tensor(X).float())
        return self.clf.predict(X_test_transform)


class HydraInternal(nn.Module):
    """HYDRA classifier."""

    def __init__(self, input_length, k=8, g=64):

        super().__init__()

        self.k = k  # num kernels per group
        self.g = g  # num groups

        max_exponent = np.log2((input_length - 1) / (9 - 1))  # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div(
            (9 - 1) * self.dilations, 2, rounding_mode="floor"
        ).int()

        # if g > 1, assign: half the groups to X, half the groups to diff(X)
        divisor = 2 if self.g > 1 else 1
        _g = g // divisor
        self._g = _g

        self.W = [
            self.normalize(torch.randn(divisor, k * _g, 1, 9))
            for _ in range(self.num_dilations)
        ]

    @staticmethod
    def normalize(W):
        """Normalize."""
        W -= W.mean(-1, keepdims=True)
        W /= W.abs().sum(-1, keepdims=True)
        return W

    # transform in batches of *batch_size*
    def batch(self, X, batch_size=256):
        """Batches."""
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for i, batch in enumerate(batches):
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X):
        """Forward."""
        num_examples = X.shape[0]

        if self.g > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            # diff_index == 0 -> X
            # diff_index == 1 -> diff(X)
            for diff_index in range(min(2, self.g)):

                _Z = F.conv1d(
                    X if diff_index == 0 else diff_X,
                    self.W[dilation_index][diff_index],
                    dilation=d,
                    padding=p,
                ).view(num_examples, self._g, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self._g, self.k)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self._g, self.k)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z
