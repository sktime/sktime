import torch
import torch.nn as nn
import numpy as np
from sktime.classification.deep_learning.base import BaseDeepClassifier

# 1. THE MATH: ResNetBlock with perfectly aligned dimensions
class ResNetBlock(nn.Module):
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
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

# 2. THE BRAIN: Connects to sktime and handles training
class ResNetClassifier(BaseDeepClassifier):
    def __init__(self, n_epochs=100, batch_size=40, random_state=None):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        super(ResNetClassifier, self).__init__()

    def _build_model(self, input_shape):
        # input_shape is (channels, length)
        model = nn.Sequential(
            ResNetBlock(input_shape[0], 64),
            ResNetBlock(64, 128),
            ResNetBlock(128, 128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, getattr(self, "n_classes_", 2))
        )
        return model

    def _fit(self, X, y):
        """The Training Loop."""
        # X shape: (n_instances, n_channels, n_timepoints)
        input_shape = (X.shape[1], X.shape[2]) 
        self.model_ = self._build_model(input_shape)
        
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.model_.train()
        for epoch in range(self.n_epochs):
            inputs = torch.from_numpy(X).float()
            labels = torch.from_numpy(y).long()
            
            optimizer.zero_grad()
            outputs = self.model_(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        return self

    def _predict(self, X):
        """Predict labels for sequences in X."""
        self.model_.eval()
        inputs = torch.from_numpy(X).float()
        
        with torch.no_grad():
            # Crucial: Call the model directly, not .predict()
            outputs = self.model_(inputs)
            predictions = torch.argmax(outputs, dim=1)
            
        return predictions.numpy()