"""Utility functions for hugging face transformers models."""

import numpy as np


def _pad_truncate(data, seq_len, pad_value=0):
    """
    Pad or truncate a numpy array.

    Parameters
    ----------
    - data: numpy array of shape (batch_size, original_seq_len, n_dims)
    - seq_len: sequence length to pad or truncate to
    - pad_value: value to use for padding

    Returns
    -------
    - padded_data: array padded or truncated to (batch_size, seq_len, n_dims)
    - mask: mask indicating padded elements (1 for existing; 0 for missing)
    """
    _, original_seq_len, _ = data.shape
    # batch_size, original_seq_len, n_dims
    # Truncate or pad each sequence in data
    if original_seq_len > seq_len:
        truncated_data = data[:, -seq_len:, :]
        mask = np.ones_like(truncated_data)
    else:
        truncated_data = np.pad(
            data,
            ((0, 0), (seq_len - original_seq_len, 0), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
        mask = np.zeros_like(truncated_data)
        mask[:, -original_seq_len:, :] = 1

    return truncated_data, mask


def _same_index(data):
    data = data.groupby(level=list(range(len(data.index.levels) - 1))).apply(
        lambda x: x.index.get_level_values(-1)
    )
    assert data.map(
        lambda x: x.equals(data.iloc[0])
    ).all(), "All series must has the same index"
    return data.iloc[0], len(data.iloc[0])


def _frame2numpy(data):
    idx, length = _same_index(data)
    arr = np.array(data.values, dtype=np.float32).reshape(
        (-1, length, len(data.columns))
    )
    return arr
