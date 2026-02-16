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

import abc
from typing import Any

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.libs.uni2ts.common.torch_util import safe_div

if _check_soft_dependencies("torch", severity="none"):
    import torch

if _check_soft_dependencies("einops", severity="none"):
    from einops import rearrange, reduce


class PackedLoss(abc.ABC):
    def __call__(
        self,
        pred: Any,
        target,
        prediction_mask,
        observed_mask,
        sample_id,
        variate_id,
    ):
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros_like(prediction_mask, dtype=torch.long)
        if variate_id is None:
            variate_id = torch.zeros_like(prediction_mask, dtype=torch.long)

        loss = self._loss_func(
            pred, target, prediction_mask, observed_mask, sample_id, variate_id
        )
        return self.reduce_loss(
            loss, prediction_mask, observed_mask, sample_id, variate_id
        )

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: Any,
        target,
        prediction_mask,
        observed_mask,
        sample_id,
        variate_id,
    ):
        pass

    def reduce_loss(
        self,
        loss,
        prediction_mask,
        observed_mask,
        sample_id,
        variate_id,
    ):
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        mask = prediction_mask.unsqueeze(-1) * observed_mask
        tobs = reduce(
            id_mask
            * reduce(
                mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        nobs = reduce(
            id_mask * rearrange(prediction_mask, "... seq -> ... 1 seq"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        ) * prediction_mask.unsqueeze(-1)
        nobs = torch.where(nobs == 0, nobs, 1 / nobs).sum()
        loss = safe_div(loss, tobs * nobs)
        return (loss * mask).sum()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PackedDistributionLoss(PackedLoss):
    @abc.abstractmethod
    def _loss_func(
        self,
        pred,
        target,
        prediction_mask,
        observed_mask,
        sample_id,
        variate_id,
    ):
        pass
