"""Test LTSF."""
import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sktime.networks.ltsf.ltsf import LTSFLinearForecaster

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train = pd.read_csv("ETT/ETTh1.csv").drop("date", axis=1)
# test_size = 0.3
# train = data.iloc[int(len(data) * test_size) :]
# test = data.iloc[: int(len(data) * test_size)]

seq_len = 336
pred_len = 96

network = LTSFLinearForecaster(
    seq_len=seq_len,
    pred_len=pred_len,
    lr=0.005,
    batch_size=32,
    in_channels=7,
    num_epochs=600,
    individual=False,
)

try:
    network.fit(train)
    y_pred = network.predict(X=train)
    y_true = network.get_y_true(train)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
except Exception:
    network.save("model.pth")

network.save("model.pth")
