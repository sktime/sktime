# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from functools import partial
from typing import Callable, Optional

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Beta, Distribution, constraints

if _check_soft_dependencies("gluonts"):
    from gluonts.core.component import validated
    from gluonts.torch.distributions import DistributionOutput
    from gluonts.torch.modules.lambda_layer import LambdaLayer


class QuantileLayer(nn.Module):
    r"""
    Implicit Quantile Layer from the paper ``IQN for Distributional
    Reinforcement Learning`` (https://arxiv.org/abs/1806.06923) by
    Dabney et al. 2018.
    """

    def __init__(self, num_output: int, cos_embedding_dim: int = 128):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(cos_embedding_dim, cos_embedding_dim),
            nn.PReLU(),
            nn.Linear(cos_embedding_dim, num_output),
        )

        self.register_buffer("integers", torch.arange(0, cos_embedding_dim))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:  # tau: [B, T]
        cos_emb_tau = torch.cos(tau.unsqueeze(-1) * self.integers * torch.pi)
        return self.output_layer(cos_emb_tau)


class ImplicitQuantileModule(nn.Module):
    r"""
    Implicit Quantile Network from the paper ``IQN for Distributional
    Reinforcement Learning`` (https://arxiv.org/abs/1806.06923) by
    Dabney et al. 2018.
    """

    def __init__(
        self,
        in_features: int,
        args_dim: dict[str, int],
        domain_map: Callable[..., tuple[torch.Tensor]],
        concentration1: float = 1.0,
        concentration0: float = 1.0,
        output_domain_map=None,
        cos_embedding_dim: int = 64,
    ):
        super().__init__()
        self.output_domain_map = output_domain_map
        self.domain_map = domain_map
        self.beta = Beta(concentration1=concentration1, concentration0=concentration0)

        self.quantile_layer = QuantileLayer(
            in_features, cos_embedding_dim=cos_embedding_dim
        )
        self.output_layer = nn.Sequential(
            nn.Linear(in_features, in_features), nn.PReLU()
        )

        self.proj = nn.ModuleList(
            [nn.Linear(in_features, dim) for dim in args_dim.values()]
        )

    def forward(self, inputs: torch.Tensor):
        if self.training:
            taus = self.beta.sample(sample_shape=inputs.shape[:-1]).to(inputs.device)
        else:
            taus = torch.rand(size=inputs.shape[:-1], device=inputs.device)

        emb_taus = self.quantile_layer(taus)
        emb_inputs = inputs * (1.0 + emb_taus)

        emb_outputs = self.output_layer(emb_inputs)
        outputs = [proj(emb_outputs).squeeze(-1) for proj in self.proj]
        if self.output_domain_map is not None:
            outputs = [self.output_domain_map(output) for output in outputs]
        return (*self.domain_map(*outputs), taus)


class ImplicitQuantileNetwork(Distribution):
    r"""
    Distribution class for the Implicit Quantile from which
    we can sample or calculate the quantile loss.

    Parameters
    ----------
    outputs
        Outputs from the Implicit Quantile Network.
    taus
        Tensor random numbers from the Beta or Uniform distribution for the
        corresponding outputs.
    """

    arg_constraints: dict[str, constraints.Constraint] = {}

    def __init__(self, outputs: torch.Tensor, taus: torch.Tensor, validate_args=None):
        self.taus = taus
        self.outputs = outputs

        super().__init__(batch_shape=outputs.shape, validate_args=validate_args)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return self.outputs

    def quantile_loss(self, value: torch.Tensor) -> torch.Tensor:
        # penalize by tau for under-predicting
        # and by 1-tau for over-predicting
        return (self.taus - (value < self.outputs).float()) * (value - self.outputs)


class ImplicitQuantileNetworkOutput(DistributionOutput):
    r"""
    DistributionOutput class for the IQN from the paper
    ``Probabilistic Time Series Forecasting with Implicit Quantile Networks``
    (https://arxiv.org/abs/2107.03743) by Gouttes et al. 2021.

    Parameters
    ----------
    output_domain
        Optional domain mapping of the output. Can be "positive", "unit"
        or None.
    concentration1
        Alpha parameter of the Beta distribution when sampling the taus
        during training.
    concentration0
        Beta parameter of the Beta distribution when sampling the taus
        during training.
    cos_embedding_dim
        The embedding dimension for the taus embedding layer of IQN.
        Default is 64.
    """

    distr_cls = ImplicitQuantileNetwork
    args_dim = {"quantile_function": 1}

    @validated()
    def __init__(
        self,
        output_domain: Optional[str] = None,
        concentration1: float = 1.0,
        concentration0: float = 1.0,
        cos_embedding_dim: int = 64,
    ) -> None:
        super().__init__()

        self.concentration1 = concentration1
        self.concentration0 = concentration0
        self.cos_embedding_dim = cos_embedding_dim

        if output_domain in ["positive", "unit"]:
            output_domain_map_func = {
                "positive": F.softplus,
                "unit": partial(F.softmax, dim=-1),
            }
            self.output_domain_map = output_domain_map_func[output_domain]
        else:
            self.output_domain_map = None

    def get_args_proj(self, in_features: int) -> nn.Module:
        return ImplicitQuantileModule(
            in_features=in_features,
            args_dim=self.args_dim,
            output_domain_map=self.output_domain_map,
            domain_map=LambdaLayer(self.domain_map),
            concentration1=self.concentration1,
            concentration0=self.concentration0,
            cos_embedding_dim=self.cos_embedding_dim,
        )

    @classmethod
    def domain_map(cls, *args):
        return args

    def distribution(self, distr_args, loc=0, scale=None) -> ImplicitQuantileNetwork:
        (outputs, taus) = distr_args

        if scale is not None:
            outputs = outputs * scale
        if loc is not None:
            outputs = outputs + loc
        return self.distr_cls(outputs=outputs, taus=taus)

    @property
    def event_shape(self):
        return ()

    def loss(
        self,
        target: torch.Tensor,
        distr_args: tuple[torch.Tensor, ...],
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        distribution = self.distribution(distr_args, loc=loc, scale=scale)
        return distribution.quantile_loss(target)


iqn = ImplicitQuantileNetworkOutput()
