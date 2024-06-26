"""Implements Pytorch Dataset class for LTSF-Formers."""

import numpy as np
import torch
from torch.utils.data import Dataset

from sktime.networks.ltsf.utils.timefeatures import time_features


class PytorchFormerDataset(Dataset):
    """Implements Pytorch Dataset class for LTSF-Formers."""

    def __init__(
        self,
        y,
        seq_len,
        context_len,
        pred_len,
        freq,
        temporal_encoding,
        temporal_encoding_type,
    ):
        self.y = y
        self.seq_len = seq_len
        self.context_len = context_len
        self.pred_len = pred_len
        self.freq = freq
        self.temporal_encoding = temporal_encoding
        self.temporal_encoding_type = temporal_encoding_type

        self._prepare_data()

    def _prepare_data(self):
        time_stamps = self.y.index
        data = self.y.values

        if self.temporal_encoding:
            if self.temporal_encoding_type == "linear":
                time_stamps = time_features(time_stamps, freq=self.freq)
                time_stamps = time_stamps.transpose(1, 0)

            elif self.temporal_encoding_type == "embed" or self.temporal_encoding_type == "fixed-embed":
                def time_stamps_map(x):
                    return [x.month, x.day, x.weekday, x.hour]
                time_stamps = np.vstack(time_stamps.map(time_stamps_map))

            else:
                raise ValueError()
        else:
            pass
            # TODO: check for temporal_encoding

        # TODO: process the time stamps
        #       - to scale the data to range (0, vocab_size), put extra to OV
        #       - pad extra columns to get fixed length

        self.time_stamps = time_stamps
        self.data = data

    def __len__(self):
        """Get length of the dataset."""
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        """Get data pairs at this index."""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.context_len
        r_end = r_begin + self.context_len + self.pred_len

        seq_x = torch.tensor(self.data[s_begin:s_end]).float()
        seq_y = torch.tensor(self.data[r_begin:r_end]).float()
        seq_x_mark = torch.tensor(self.time_stamps[s_begin:s_end]).float()
        seq_y_mark = torch.tensor(self.time_stamps[r_begin:r_end]).float()

        dec_inp = torch.zeros_like(seq_y[-self.pred_len :, :])
        dec_inp = torch.cat([seq_y[: self.context_len, :], dec_inp], dim=0)

        y_true = seq_y[-self.pred_len :]

        return (
            {
                "x_enc": seq_x,
                "x_mark_enc": seq_x_mark,
                "x_dec": dec_inp,
                "x_mark_dec": seq_y_mark,
            },
            y_true,
        )
