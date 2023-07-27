"""Test LTSF."""
import random

import pandas as pd

from sktime.forecasting.ltsf import LTSFLinearForecaster


def train_test_split(dataset_path, input_output_len, test_size, col_name="HUFL"):
    """Shuffle data for LTSF-Linear."""
    data = pd.read_csv(dataset_path).drop("date", axis=1)
    # data = [
    #     data[i : i + input_output_len] for i in range(
    #         0, len(data), input_output_len
    #     ) if len(data[i : i + input_output_len]) == input_output_len
    # ]
    cutoff = int(len(data) * test_size)
    # data = pd.DataFrame(data)
    indicies = list(range(len(data)))
    random.shuffle(indicies)
    # get rows
    train = data[:, indicies[:cutoff]].reset_index()
    test = data[indicies[cutoff:]].reset_index()
    return train, test


train, test = train_test_split("examples/ltsf/ETT/ETTh1.csv", 96 + 1, 0.2)
# train, test = train_test_split("ETT/ETTh1.csv", 96 + 1, 0.2)

network = LTSFLinearForecaster(seq_len=96, pred_len=1, num_epochs=100)
network.fit(train, fh=[1])
