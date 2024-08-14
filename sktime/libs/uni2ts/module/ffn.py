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

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * in_dim
        out_dim = out_dim or in_dim

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bias = bias
        self.ffn_dropout_p = ffn_dropout_p

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout1 = nn.Dropout(ffn_dropout_p)
        self.dropout2 = nn.Dropout(ffn_dropout_p)
        self.activation = activation

    def forward(
        self, x: Float[torch.Tensor, "... in_dim"]
    ) -> Float[torch.Tensor, "... out_dim"]:
        x = self._in_proj(x)
        return self.dropout2(self.fc2(self.dropout1(x)))

    def _in_proj(
        self, x: Float[torch.Tensor, "... in_dim"]
    ) -> Float[torch.Tensor, "... out_dim"]:
        return self.activation(self.fc1(x))


class GatedLinearUnitFeedForward(FeedForward):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__(
            in_dim,
            hidden_dim=hidden_dim or self.adjust_hidden_dim(4 * in_dim),
            out_dim=out_dim,
            activation=activation,
            bias=bias,
            ffn_dropout_p=ffn_dropout_p,
        )
        self.fc_gate = nn.Linear(self.in_dim, self.hidden_dim, bias=self.bias)

    @staticmethod
    def adjust_hidden_dim(dim):
        return (int(dim * 2 / 3) + 7) // 8 * 8

    def _in_proj(
        self, x: Float[torch.Tensor, "... in_dim"]
    ) -> Float[torch.Tensor, "... out_dim"]:
        return self.activation(self.fc_gate(x)) * self.fc1(x)
