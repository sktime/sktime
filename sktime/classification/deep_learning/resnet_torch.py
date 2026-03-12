import torch
import torch.nn as nn
from sktime.classification.deep_learning.base import BaseDeepClassifier

# 1. THE MATH: ResNetBlock with perfectly aligned dimensions
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # kernel=7, padding=3 keeps the length the same (50 -> 50)
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
        # This addition now works because dimensions match perfectly
        out += residual
        return self.relu(out)

# 2. THE BRAIN: Connects to sktime
class ResNetClassifier(BaseDeepClassifier):
    def __init__(self, n_epochs=100, batch_size=40, random_state=None):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        super(ResNetClassifier, self).__init__()

    def _build_model(self, input_shape):
        # input_shape is (channels, length)
        self.model = nn.Sequential(
            ResNetBlock(input_shape[0], 64),
            ResNetBlock(64, 128),
            ResNetBlock(128, 128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, getattr(self, "n_classes_", 2))
        )
        return self.model

# 3. THE TEST: Verification
if __name__ == "__main__":
    clf = ResNetClassifier(n_epochs=1)
    clf.n_classes_ = 3 
    
    # Test with 50 time steps
    dummy_input = torch.randn(1, 1, 50) 
    torch_model = clf._build_model(input_shape=(1, 50))
    
    # Try forward pass
    output = torch_model(dummy_input)
    
    print("\n" + "⭐"*15)
    print("✅ ABSOLUTE SUCCESS!")
    print(f"✅ Input: 50 steps -> Output: {output.shape}")
    print("⭐"*15 + "\n")