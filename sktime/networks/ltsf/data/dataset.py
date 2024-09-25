"""Implements Pytorch Dataset class for LTSF-Formers."""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class PytorchFormerDataset(Dataset):
    """Implements Pytorch Dataset class for LTSF-Formers."""

    def __init__(
        self,
        data,
        index,
        seq_len,
        context_len,
        pred_len,
        freq,
        temporal_encoding,
        temporal_encoding_type,
    ):
        self.data = data
        self.index = index
        self.seq_len = seq_len
        self.context_len = context_len
        self.pred_len = pred_len
        self.freq = freq
        self.temporal_encoding = temporal_encoding
        self.temporal_encoding_type = temporal_encoding_type

        self._num, self._len, _ = self.data.shape
        self._len_single = self._len - self.seq_len - self.pred_len + 1

        self.time_stamps = self._get_time_stamps()

    def _get_time_stamps(self):
        from sktime.networks.ltsf.utils.timefeatures import (
            generate_temporal_features,
        )

        return (
            generate_temporal_features(
                index=self.index,
                temporal_encoding_type=self.temporal_encoding_type,
                freq=self.freq,
            )
            if self.temporal_encoding
            else None
        )

    def __len__(self):
        """Return length of dataset."""
        return self._num * max(self._len_single, 0)

    def __getitem__(self, index):
        """Get data pairs at this index."""
        n = index // self._len_single
        m = index % self._len_single

        s_begin = m  # m
        s_end = s_begin + self.seq_len  # m+seq
        r_begin = s_end - self.context_len  # m+seq-context
        r_end = r_begin + self.context_len + self.pred_len  # m+seq+pred

        seq_x = self.data[n, s_begin:s_end]
        seq_y = self.data[n, r_begin:r_end]

        seq_x = torch.tensor(seq_x).float()
        seq_y = torch.tensor(seq_y).float()

        if self.temporal_encoding:
            seq_x_mark = torch.tensor(self.time_stamps[s_begin:s_end]).float()
            seq_y_mark = torch.tensor(self.time_stamps[r_begin:r_end]).float()
        else:
            seq_x_mark = torch.empty((self.seq_len, 0)).float()
            seq_y_mark = torch.empty((self.context_len + self.pred_len, 0)).float()

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
