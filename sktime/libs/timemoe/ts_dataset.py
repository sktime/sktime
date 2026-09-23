"""Time series dataset base class for Time-MoE training."""

from abc import abstractmethod


class TimeSeriesDataset:
    """Abstract base for Time-MoE training datasets."""

    @abstractmethod
    def __len__(self):
        """Return number of sequences."""

    @abstractmethod
    def __getitem__(self, seq_idx):
        """Return sequence at ``seq_idx``."""

    @abstractmethod
    def get_num_tokens(self):
        """Return total number of tokens across sequences."""

    @abstractmethod
    def get_sequence_length_by_idx(self, seq_idx):
        """Return length of sequence at ``seq_idx``."""

    @staticmethod
    def is_valid_path(data_path):
        """Return whether ``data_path`` is a valid dataset path."""
        return True

    def __iter__(self):
        """Iterate over sequences."""
        n_seqs = len(self)
        for i in range(n_seqs):
            yield self[i]


class SeriesListDataset(TimeSeriesDataset):
    """Wrap a list of 1D numpy series as a ``TimeSeriesDataset``."""

    def __init__(self, series_list):
        self.series_list = list(series_list)

    def __len__(self):
        """Return number of sequences."""
        return len(self.series_list)

    def __getitem__(self, seq_idx):
        """Return sequence at ``seq_idx``."""
        return self.series_list[seq_idx]

    def get_num_tokens(self):
        """Return total number of tokens across sequences."""
        return sum(len(series) for series in self.series_list)

    def get_sequence_length_by_idx(self, seq_idx):
        """Return length of sequence at ``seq_idx``."""
        return len(self.series_list[seq_idx])
