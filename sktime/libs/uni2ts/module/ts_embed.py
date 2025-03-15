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

import math

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch import nn
else:

    class nn:
        class Module:
            pass


if _check_soft_dependencies("einops", severity="none"):
    from einops import einsum, rearrange

from sktime.libs.uni2ts.common.torch_util import size_to_mask


def fs2idx(feat_size, feat_sizes):
    return (
        (rearrange(feat_size, "... -> ... 1") == feat_sizes)
        .to(torch.long)
        .argmax(dim=-1)
    )


class MultiInSizeLinear(nn.Module):
    def __init__(
        self,
        in_features_ls: tuple[int, ...],
        out_features: int,
        bias: bool = True,
        dtype=None,
    ):
        super().__init__()
        self.in_features_ls = in_features_ls
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                (len(in_features_ls), out_features, max(in_features_ls)), dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(in_features_ls), out_features), dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.register_buffer(
            "mask",
            rearrange(
                size_to_mask(max(in_features_ls), torch.as_tensor(in_features_ls)),
                "num_feats max_feat -> num_feats 1 max_feat",
            ),
            persistent=False,
        )
        self.register_buffer(
            "in_features_buffer",
            torch.tensor(in_features_ls),
            persistent=False,
        )

    def reset_parameters(self):
        for idx, feat_size in enumerate(self.in_features_ls):
            nn.init.kaiming_uniform_(self.weight[idx, :, :feat_size], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, :, feat_size:])
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[idx, :, :feat_size]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[idx], -bound, bound)

    def forward(
        self,
        x,
        in_feat_size,
    ):
        out = 0
        for idx, feat_size in enumerate(self.in_features_ls):
            weight = self.weight[idx] * self.mask[idx]
            bias = self.bias[idx] if self.bias is not None else 0
            out = out + (
                torch.eq(in_feat_size, feat_size).unsqueeze(-1)
                * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)
            )
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features_ls={self.in_features_ls}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.weight.dtype}"
        )


class MultiOutSizeLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features_ls: tuple[int, ...],
        dim: int = 1,
        bias: bool = True,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_ls = out_features_ls
        self.dim = dim

        self.weight = nn.Parameter(
            torch.empty(
                (len(out_features_ls), max(out_features_ls), in_features), dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(out_features_ls), max(out_features_ls)), dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.register_buffer(
            "mask",
            rearrange(
                size_to_mask(max(out_features_ls), torch.as_tensor(out_features_ls)),
                "num_feats max_feat -> num_feats max_feat 1",
            ),
            persistent=False,
        )
        self.register_buffer(
            "out_features_buffer",
            torch.tensor(out_features_ls),
            persistent=False,
        )

    def reset_parameters(self):
        for idx, feat_size in enumerate(self.out_features_ls):
            nn.init.kaiming_uniform_(self.weight[idx, :feat_size], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, feat_size:])
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[idx, :feat_size]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[idx, :feat_size], -bound, bound)
                nn.init.zeros_(self.bias[idx, feat_size:])

    def forward(
        self,
        x,
        out_feat_size,
    ):
        out = 0
        for idx, feat_size in enumerate(self.out_features_ls):
            weight = self.weight[idx] * self.mask[idx]
            bias = self.bias[idx] if self.bias is not None else 0
            out = out + (
                torch.eq(out_feat_size, feat_size // self.dim).unsqueeze(-1)
                * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)
            )
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features_ls={self.out_features_ls}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.weight.dtype}"
        )
