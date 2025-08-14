# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import datasets
import torch

from .standard_adapter import _batch_pad_iterable

DEF_TARGET_COLUMN = "target"


def _get_hf_map(dataset: datasets.Dataset, **hf_kwargs):
    target_col = hf_kwargs.get("target_column", DEF_TARGET_COLUMN)
    meta_columns = hf_kwargs.get("meta_columns", ())

    columns_to_pass = [target_col] + list(meta_columns)
    remove_cols = [col for col in dataset.column_names if col not in columns_to_pass]
    dataset = (
        dataset.with_format("torch")
        .remove_columns(remove_cols)
        .cast_column(target_col, datasets.Sequence(datasets.Value("float32")))
    )

    def yield_batch_tuples(sample: dict) -> tuple[torch.Tensor, dict]:
        context_data = sample[target_col]
        if context_data.ndim > 1:
            context_data = context_data.squeeze()
        assert context_data.ndim == 1
        meta = {k: sample[k] for k in meta_columns if k in sample}
        meta["length"] = len(context_data)
        return context_data, meta

    return dataset, yield_batch_tuples


def get_hfdata_batches(hf_dataset: datasets.Dataset, batch_size: int, **hf_kwargs):
    dataset, map_func = _get_hf_map(hf_dataset, **hf_kwargs)
    return _batch_pad_iterable(map(map_func, dataset), batch_size)
