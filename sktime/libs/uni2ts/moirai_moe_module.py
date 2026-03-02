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

from functools import partial

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn.functional as F
    from torch import nn
    from torch.utils._pytree import tree_map

    from .distribution import DistributionOutput
    from .module.norm import RMSNorm
    from .module.position import (
        BinaryAttentionBias,
        QueryKeyProjection,
        RotaryProjection,
    )
    from .module.transformer import TransformerEncoder
    from .module.ts_embed import FeatLinear, MultiInSizeLinear
else:

    class nn:
        class Module:
            pass

    class DistributionOutput:
        pass


if _check_soft_dependencies("huggingface_hub", severity="none"):
    from huggingface_hub import PyTorchModelHubMixin
else:

    class PyTorchModelHubMixin:
        def __init__(self):
            pass

        def __init_subclass__(cls, *args, **kwargs) -> None:
            """Implement dummy version of __init_subclass__."""
            pass


if _check_soft_dependencies("einops", severity="none"):
    from .module.packed_scaler import PackedNOPScaler, PackedStdScaler

from .common.torch_util import packed_causal_attention_mask


def encode_distr_output(
    distr_output: DistributionOutput,
):
    """Serialize function for DistributionOutput."""

    def _encode(val):
        if not isinstance(val, DistributionOutput):
            return val

        return {
            "_target_": f"{val.__class__.__module__}.{val.__class__.__name__}",
            **tree_map(_encode, val.__dict__),
        }

    return _encode(distr_output)


def decode_distr_output(config) -> DistributionOutput:
    """Deserialize function for DistributionOutput."""
    from hydra.utils import instantiate

    return instantiate(config, _convert_="all")


class MoiraiMoEModule(
    nn.Module,
    PyTorchModelHubMixin,
    coders={DistributionOutput: (encode_distr_output, decode_distr_output)},
):
    def __init__(
        self,
        distr_output: DistributionOutput,
        d_model: int,
        d_ff: int,
        num_layers: int,
        patch_sizes: tuple,  # tuple[int, ...]
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_sizes = patch_sizes
        self.max_seq_len = max_seq_len
        self.scaling = scaling

        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.in_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.res_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.feat_proj = FeatLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_moe=True,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=d_ff,
        )
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_param_proj(d_model, patch_sizes)

    def forward(
        self,
        target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
        patch_size,
    ):
        """Define the forward pass of MoiraiMoEModule."""
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale

        in_reprs = self.in_proj(scaled_target, patch_size)
        in_reprs = F.silu(in_reprs)
        in_reprs = self.feat_proj(in_reprs, patch_size)
        res_reprs = self.res_proj(scaled_target, patch_size)
        reprs = in_reprs + res_reprs

        reprs = self.encoder(
            reprs,
            packed_causal_attention_mask(sample_id, time_id),
            time_id=time_id,
            var_id=variate_id,
        )
        distr_param = self.param_proj(reprs, patch_size)
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
        return distr
