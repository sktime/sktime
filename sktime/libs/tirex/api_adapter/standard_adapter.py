# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import itertools
from collections.abc import Iterable, Iterator, Sequence
from typing import Union, Optional
import numpy as np

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")

ContextType = Union[
    torch.Tensor,
    np.ndarray,
    list[torch.Tensor],
    list[np.ndarray],
]


def _batched_slice(
    full_batch, full_meta: list[dict] | None, batch_size: int
) -> Iterator[tuple[Sequence, list[dict]]]:
    if len(full_batch) <= batch_size:
        yield (
            full_batch,
            full_meta
            if full_meta is not None
            else [{} for _ in range(len(full_batch))],
        )
    else:
        for i in range(0, len(full_batch), batch_size):
            batch = full_batch[i : i + batch_size]
            yield (
                batch,
                (
                    full_meta[i : i + batch_size]
                    if full_meta is not None
                    else [{} for _ in range(len(batch))]
                ),
            )


def _batched(iterable: Iterable, n: int):
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def _batch_pad_iterable(iterable: Iterable[tuple[torch.Tensor, dict]], batch_size: int):
    for batch in _batched(iterable, batch_size):
        # ctx_it_len, ctx_it_data, it_meta = itertools.tee(batch, 3)
        max_len = max(len(el[0]) for el in batch)
        padded_batch = []
        meta = []
        for el in batch:
            sample = el[0]
            assert isinstance(sample, torch.Tensor)
            assert sample.ndim == 1
            assert len(sample) > 0, "Each sample needs to have a length > 0"
            padding = torch.full(
                size=(max_len - len(sample),),
                fill_value=torch.nan,
                device=sample.device,
            )
            padded_batch.append(torch.cat((padding, sample)))
            meta.append(el[1])
        yield torch.stack(padded_batch), meta


def get_batches(context: ContextType, batch_size: int):
    batches = None
    if isinstance(context, torch.Tensor):
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2
        batches = _batched_slice(context, None, batch_size)
    elif isinstance(context, np.ndarray):
        if context.ndim == 1:
            context = np.expand_dims(context, axis=0)
        assert context.ndim == 2
        batches = map(
            lambda x: (torch.Tensor(x[0]), x[1]),
            _batched_slice(context, None, batch_size),
        )
    elif isinstance(context, (list, Iterable)):
        batches = _batch_pad_iterable(
            map(lambda x: (torch.Tensor(x), None), context), batch_size
        )
    if batches is None:
        raise ValueError(
            f"Context type {type(context)} not supported! Supported Types: {ContextType}"
        )
    return batches
