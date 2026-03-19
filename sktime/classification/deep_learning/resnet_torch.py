"""PyTorch implementation of ResNet for Time Series Classification."""

import numpy as np
import torch
import torch.nn as nn

from sktime.classification.deep_learning.base import BaseDeepClassifier


class ResNetBlock(nn.Module):
    """Residual block with perfectly aligned dimensions.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        """Forward pass for the ResNet block."""
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResNetClassifier(BaseDeepClassifier):
    """Time series classifier using Residual Neural Networks.

    Parameters
    ----------
    n_epochs : int, default=100
        Number of epochs to train the model.
    batch_size : int, default=40
        Size of training batches.
    random_state : int or None, default=None
        Seed for reproducibility.
    """

    def __init__(self, n_epochs=100, batch_size=40, random_state=None):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        super().__init__()

    def _build_model(self, input_shape):
        """Build the PyTorch model architecture.

        Parameters
        ----------
        input_shape : tuple
            The shape of the input data (channels, length).

        Returns
        -------
        model : nn.Module
            The ResNet model architecture.
        """
        model = nn.Sequential(
            ResNetBlock(input_shape[0], 64),
            ResNetBlock(64, 128),
            ResNetBlock(128, 128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, getattr(self, "n_classes_", 2)),
        )
        return model

    def _fit(self, X, y):
        """Fit the classifier with hardware acceleration and seeding.

        Parameters
        ----------
        X : np.ndarray
            The training input samples.
        y : np.ndarray
            The training target values.

        Returns
        -------
        self : reference to self
        """
        if torch.backends.mps.is_available():
            self.device_ = torch.device("mps")
        elif torch.cuda.is_available():
            self.device_ = torch.device("cuda")
        else:
            self.device_ = torch.device("cpu")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        input_shape = (X.shape[1], X.shape[2])
        self.model_ = self._build_model(input_shape)
        self.model_.to(self.device_)

        self.model_ = torch.compile(self.model_)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for _ in range(self.n_epochs):
            inputs = torch.from_numpy(X).float().to(self.device_)
            labels = torch.from_numpy(y).long().to(self.device_)

            optimizer.zero_grad()
            outputs = self.model_(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        return self

    def _predict(self, X):
        """Predict labels with hardware acceleration.

        Parameters
        ----------
        X : np.ndarray
            The input samples for prediction.

        Returns
        -------
        y : np.ndarray
            The predicted class labels.
        """
        self.model_.eval()
        inputs = torch.from_numpy(X).float().to(self.device_)

        with torch.no_grad():
            outputs = self.model_(inputs)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy()
