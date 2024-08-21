#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch

    numpy_to_torch_dtype_dict = {
        bool: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
    }


# noqa: F722
def packed_attention_mask(
    sample_id,
):
    """
    Create a packed attention mask for self-attention.

    Parameters
    ----------
    sample_id: Int[torch.Tensor, "*batch seq_len"]

    Returns
    -------
    Bool[torch.Tensor, "*batch seq_len seq_len"]
    """
    sample_id = sample_id.unsqueeze(-1)
    attention_mask = sample_id.eq(sample_id.mT)
    return attention_mask


def mask_fill(
    tensor,
    mask,
    value,
):
    """
    Fill masked values with a given value.

    Parameters
    ----------
    tensor: Float[torch.Tensor, "*batch dim"]
    mask: Bool[torch.Tensor, "*batch"]
    value: Float[torch.Tensor, "dim"]

    Returns
    -------
    Float[torch.Tensor, "*batch dim"]

    """
    mask = mask.unsqueeze(-1)
    return tensor * ~mask + value * mask


def safe_div(
    numer,
    denom,
):
    """
    Safe division.

    Parameters
    ----------
    numer: torch.Tensor
    denom: torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    return numer / torch.where(
        denom == 0,
        1.0,
        denom,
    )


def size_to_mask(
    max_size: int,
    sizes,
):
    """
    Create a mask for a given size.

    Parameters
    ----------
    max_size: int
    sizes: Int[torch.Tensor, "*batch"]

    Returns
    -------
    Bool[torch.Tensor, "*batch max_size"]
    """
    mask = torch.arange(max_size, device=sizes.device)
    return torch.lt(mask, sizes.unsqueeze(-1))


def unsqueeze_trailing_dims(x, shape):
    """
    Unsqueeze trailing dimensions.

    Parameters
    ----------
    x: torch.Tensor
    shape: torch.Size

    Returns
    -------
    torch.Tensor

    """
    if x.ndim > len(shape) or x.shape != shape[: x.ndim]:
        raise ValueError
    dim = (...,) + (None,) * (len(shape) - x.ndim)
    return x[dim]
