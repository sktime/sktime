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

from typing import Callable, Optional

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.distributions import Normal
    from torch.nn import functional as F

else:
    # Create Dummy class
    class Normal:
        pass


from ._base import DistributionOutput


class NormalOutput(DistributionOutput):
    distr_cls = Normal
    args_dim = dict(loc=1, scale=1)

    @property
    def domain_map(self) -> [Callable, "T"]:
        return dict(
            loc=self._loc,
            scale=self._scale,
        )

    @staticmethod
    def _loc(loc):
        return loc.squeeze(-1)

    @staticmethod
    def _scale(scale):
        epsilon = torch.finfo(scale.dtype).eps
        return F.softplus(scale).clamp_min(epsilon).squeeze(-1)


class NormalFixedScaleOutput(DistributionOutput):
    distr_cls = Normal
    args_dim = dict(loc=1)

    def __init__(self, scale: float = 1e-3):
        self.scale = scale

    @property
    def domain_map(
        self,
    ):
        return dict(loc=self._loc)

    @staticmethod
    def _loc(loc):
        return loc.squeeze(-1)

    def _distribution(
        self,
        distr_params,
        validate_args: Optional[bool] = None,
    ) -> Normal:
        loc = distr_params["loc"]
        distr_params["scale"] = torch.as_tensor(
            self.scale, dtype=loc.dtype, device=loc.device
        )
        return self.distr_cls(**distr_params, validate_args=validate_args)
