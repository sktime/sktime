#!/usr/bin/env python
from abc import abstractmethod


class TimeSeriesDataset:
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, seq_idx):
        pass

    @abstractmethod
    def get_sequence_length_by_idx(self, seq_idx):
        pass

    @staticmethod
    @abstractmethod
    def is_valid_path(data_path):
        pass
