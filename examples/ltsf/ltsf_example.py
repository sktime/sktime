"""Test LTSF."""
import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sktime.forecasting.ltsf import LTSFLinearForecaster

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv("ETT/ETTh1.csv").drop("date", axis=1)
test_size = 0.3
train = data.iloc[int(len(data) * test_size) :]
test = data.iloc[: int(len(data) * test_size)]

seq_len = 96
pred_len = 96

network = LTSFLinearForecaster(
    seq_len=96,
    pred_len=1,
    lr=0.001,
    batch_size=32,
    in_channels=7,
    num_epochs=200,
)
network.fit(train)
y_pred = network.predict(X=test)
y_true = network.get_y_true(test)

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
