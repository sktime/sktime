"""Test LTSF."""
import os

import pandas as pd
from data_loader import Dataset_ETT_hour, Dataset_Pred
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sktime.forecasting.ltsf import LTSFLinearForecaster

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train = pd.read_csv("ETT/ETTh1.csv").drop("date", axis=1)

dataset_kwargs = {
    "features": "M",
}

train_dataset = Dataset_ETT_hour(**dataset_kwargs)
pred_dataset = Dataset_Pred(**dataset_kwargs)

network_kwargs = {
    "seq_len": 384,
    "pred_len": 192,
    "target": "OT",
    "features": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
    "lr": 0.005,
    "batch_size": 32,
    "in_channels": 7,
    "num_epochs": 600,
    "custom_dataset_train": train_dataset,
    "custom_dataset_pred": pred_dataset,
    "individual": True,
}

network = LTSFLinearForecaster(**network_kwargs)

network.fit(train)
y_pred = network.predict(X=train)
y_true = network.get_y_true(train)

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

# print(f"mse: {mse}, mae: {mae}")

network.save("model.pth")
