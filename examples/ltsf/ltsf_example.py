"""Test LTSF."""
import random

import pandas as pd
import torch

from sktime.forecasting.ltsf import LTSFLinearForecaster


def train_test_split(dataset_path, input_output_len, test_size):
    """Shuffle data for LTSF-Linear."""
    data = pd.read_csv(dataset_path).drop("date", axis=1)
    data = [
        data[i : i + input_output_len] for i in range(0, len(data), input_output_len)
    ]
    data = [x for x in data]
    random.shuffle(data)
    cutoff = int(len(data) * test_size)
    train = data[cutoff:]
    test = data[:cutoff]
    return train, test


class ETTDataset:
    """Dataset for use in LTSFLinearForecaster."""

    def __init__(self, data):
        self.data = data
        self.cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    def __len__(self):
        """Length of dataset."""
        return len(self.data)

    def __getitem__(self, i):
        """Get (x, y)."""
        return (
            torch.from_numpy(self.data[i].iloc[:-1][self.cols].values).float(),
            torch.from_numpy(self.data[i].iloc[-1][self.cols].values).float()[None],
        )


train, test = train_test_split("examples_dl/ETT/ETTh1.csv", 96 + 1, 0.2)
train_dataset = ETTDataset(train)
test_dataset = ETTDataset(test)

network = LTSFLinearForecaster(seq_len=96, pred_len=1, num_epochs=100)
network.fit(train_dataset)
