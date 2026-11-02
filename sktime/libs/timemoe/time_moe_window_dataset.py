"""Sliding-window datasets for Time-MoE training."""

import random

from sktime.libs.timemoe.ts_dataset import TimeSeriesDataset
from sktime.utils.dependencies import _safe_import

np = _safe_import("numpy")


class TimeMoEWindowDataset:
    """Non-overlapping sliding windows for Time-MoE training.

    Generates fixed-length input sequences and corresponding labels from an
    underlying ``TimeSeriesDataset``.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        context_length: int,
        prediction_length: int = 0,
        stride: int = None,
        **kwrags,
    ):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1
        self.stride = stride if stride else self.window_size

        num_seqs = len(self.dataset)
        iterator = range(num_seqs)
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=num_seqs)
        except ImportError:
            pass
        self.sub_seq_indexes = []
        for seq_idx in iterator:
            n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
            # Skip sequences with fewer than 2 points
            if n_points < 2:
                continue
            self.sub_seq_indexes.append((seq_idx, 0))
            for offset_idx in range(
                self.stride, n_points - self.window_size_plus_one + 1, self.stride
            ):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        """Return number of sliding windows."""
        return len(self.sub_seq_indexes)

    def __iter__(self):
        """Iterate over sliding windows."""
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, seq_idx):
        """Return one window as input_ids, labels, and loss_masks."""
        seq_i, offset_i = self.sub_seq_indexes[seq_idx]
        seq = self.dataset[seq_i][offset_i : offset_i + self.window_size_plus_one]
        seq = np.array(seq, dtype=np.float32)

        loss_mask = np.ones(len(seq) - 1, dtype=np.int32)
        n_pad = self.window_size_plus_one - len(seq)
        if n_pad > 0:
            seq = np.pad(seq, (0, n_pad), "constant", constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), "constant", constant_values=0)

        return {
            "input_ids": seq[:-1],
            "labels": seq[1:],
            "loss_masks": loss_mask,
        }


class UniversalTimeMoEWindowDataset:
    """Window dataset that packs sequences into fixed-size windows."""

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        context_length: int,
        prediction_length: int = 0,
        shuffle: bool = False,
        **kwrags,
    ):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length

        self.window_info_list = []
        n_seqs = len(self.dataset)

        cur_window_info = []
        num_cur_remaining_points = self.window_size

        iterator = range(n_seqs)
        if shuffle:
            iterator = list(iterator)
            random.shuffle(iterator)

        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=n_seqs)
        except ImportError:
            pass

        for seq_idx in iterator:
            seq_len = self.dataset.get_sequence_length_by_idx(seq_idx)
            remaining_seq_len = seq_len
            while remaining_seq_len > 0:
                if remaining_seq_len < num_cur_remaining_points:
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, remaining_seq_len)
                    )

                    # update states
                    num_cur_remaining_points -= remaining_seq_len
                    remaining_seq_len = 0
                else:
                    # add the part of this seq to cur_window
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, num_cur_remaining_points)
                    )

                    # update states
                    remaining_seq_len -= num_cur_remaining_points
                    self.window_info_list.append(cur_window_info)

                    # reset current window
                    num_cur_remaining_points = self.window_size
                    cur_window_info = []

        if num_cur_remaining_points > 0:
            # drop last batch for speed-up
            pass

    def __len__(self):
        """Return number of packed windows."""
        return len(self.window_info_list)

    def __getitem__(self, window_idx):
        """Return one packed window as input_ids and labels."""
        window_info = self.window_info_list[window_idx]
        seq = []
        for seq_idx, start_idx_in_seq, offset in window_info:
            part_seq = self.dataset[seq_idx][
                start_idx_in_seq : start_idx_in_seq + offset
            ]
            seq.append(part_seq)
        if len(seq) == 1:
            seq = seq[0]
            if not isinstance(seq, np.ndarray):
                seq = np.array(seq, dtype=np.float32)
            else:
                seq = seq.astype(np.float32)
        else:
            seq = np.concatenate(seq, axis=0, dtype=np.float32)
        return {
            "input_ids": seq[:-1],
            "labels": seq[1:],
        }
