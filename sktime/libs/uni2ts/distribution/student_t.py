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

from typing import Callable

import torch
from jaxtyping import Float, PyTree
from torch.distributions import StudentT
from torch.nn import functional as F

from ._base import DistributionOutput


class StudentTOutput(DistributionOutput):
    distr_cls = StudentT
    args_dim = dict(df=1, loc=1, scale=1)

    @property
    def domain_map(
        self,
    ) -> PyTree[
        Callable[[Float[torch.Tensor, "*batch 1"]], Float[torch.Tensor, "*batch"]], "T"
    ]:
        return dict(df=self._df, loc=self._loc, scale=self._scale)

    @staticmethod
    def _df(df: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        return (2.0 + F.softplus(df)).squeeze(-1)

    @staticmethod
    def _loc(loc: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        return loc.squeeze(-1)

    @staticmethod
    def _scale(scale: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        epsilon = torch.finfo(scale.dtype).eps
        return F.softplus(scale).clamp_min(epsilon).squeeze(-1)
