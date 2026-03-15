from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
import torch
import torch.nn as nn
import pandas as pd

class DummyModel(nn.Module):
    def __init__(self, seq_len=1, pred_len=1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        return self.linear(x)

class Forecaster(BaseDeepNetworkPyTorch):
    def __init__(self, model, num_epochs=16, batch_size=8, lr=0.001):
        self.model = model
        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
        )

    def _build_network(self, fh):
        return self.model

model = DummyModel(seq_len=5, pred_len=3)
forecaster = Forecaster(model=model)
forecaster.fit(y=pd.Series(range(20)), fh=[1, 2, 3])
print("Script executed successfully without AttributeError!")
