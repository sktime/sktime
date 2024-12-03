#!/usr/bin/env python
import os

from torch.utils.data import Dataset

from .binary_dataset import BinaryDataset
from .general_dataset import GeneralDataset


class TimeMoEDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.datasets = []

        if BinaryDataset.is_valid_path(self.data_folder):
            ds = BinaryDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        elif GeneralDataset.is_valid_path(self.data_folder):
            ds = GeneralDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        else:
            # walk through the data_folder
            for root, dirs, files in os.walk(self.data_folder):
                for file in files:
                    fn_path = os.path.join(root, file)
                    if (
                        file != BinaryDataset.meta_file_name
                        and GeneralDataset.is_valid_path(fn_path)
                    ):
                        ds = GeneralDataset(fn_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)
                for sub_folder in dirs:
                    folder_path = os.path.join(root, sub_folder)
                    if BinaryDataset.is_valid_path(folder_path):
                        ds = BinaryDataset(folder_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)

        self.cumsum_lengths = [0]
        for ds in self.datasets:
            self.cumsum_lengths.append(self.cumsum_lengths[-1] + len(ds))
        self.num_sequences = self.cumsum_lengths[-1]

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        if seq_idx >= self.cumsum_lengths[-1]:
            raise ValueError(
                f"Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}"
            )
        elif seq_idx < 0:
            raise ValueError(f"Index out of the dataset length: {seq_idx} < 0")

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        print(dataset_idx, dataset_offset, self.cumsum_lengths[dataset_idx])
        return self.datasets[dataset_idx][dataset_offset]


def binary_search(sorted_list, value):
    low = 0
    high = len(sorted_list) - 1
    best_index = -1

    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] <= value:
            best_index = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_index
