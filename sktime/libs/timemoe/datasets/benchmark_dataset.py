#!/usr/bin/env python
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class BenchmarkEvalDataset(Dataset):
    def __init__(self, csv_path, context_length: int, prediction_length: int):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        df = pd.read_csv(csv_path)

        base_name = os.path.basename(csv_path).lower()
        if "etth" in base_name:
            border1s = [
                0,
                12 * 30 * 24 - context_length,
                12 * 30 * 24 + 4 * 30 * 24 - context_length,
            ]
            border2s = [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ]
        elif "ettm" in base_name:
            border1s = [
                0,
                12 * 30 * 24 * 4 - context_length,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - context_length,
            ]
            border2s = [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]
        else:
            num_train = int(len(df) * 0.7)
            num_test = int(len(df) * 0.2)
            num_vali = len(df) - num_train - num_test
            border1s = [
                0,
                num_train - context_length,
                len(df) - num_test - context_length,
            ]
            border2s = [num_train, num_train + num_vali, len(df)]

        start_dt = df.iloc[border1s[2]]["date"]
        eval_start_dt = df.iloc[border1s[2] + context_length]["date"]
        end_dt = df.iloc[border2s[2] - 1]["date"]
        print(
            f">>> Split test data from {start_dt} to {end_dt}, "
            f"and evaluation start date is: {eval_start_dt}"
        )

        cols = df.columns[1:]
        df_values = df[cols].values

        train_data = df_values[border1s[0] : border2s[0]]
        test_data = df_values[border1s[2] : border2s[2]]

        # scaling
        scaler = StandardScaler()
        scaler.fit(train_data)
        scaled_test_data = scaler.transform(test_data)

        # assignment
        self.hf_dataset = scaled_test_data.transpose(1, 0)
        self.num_sequences = len(self.hf_dataset)
        # 1 for the label
        self.window_length = self.context_length + self.prediction_length

        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.hf_dataset[seq_i]

        window_seq = np.array(
            seq[offset_i - self.window_length : offset_i], dtype=np.float32
        )

        return {
            "inputs": np.array(window_seq[: self.context_length], dtype=np.float32),
            "labels": np.array(window_seq[-self.prediction_length :], dtype=np.float32),
        }
