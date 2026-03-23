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
    import torch
    import torch.nn.functional as F
    from torch import nn

    from .module.norm import RMSNorm
    from .module.position import (
        BinaryAttentionBias,
        QueryKeyProjection,
        RotaryProjection,
    )
    from .module.transformer import TransformerEncoder
    from .module.ts_embed import ResidualBlock
else:

    class nn:
        class Module:
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


class Moirai2Module(
    nn.Module,
    PyTorchModelHubMixin,
):
    """Contains components of Moirai 2.0 decoder-only forecaster."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_layers: int,
        patch_size: int,
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
        num_predict_token: int = 1,
        quantile_levels: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_predict_token = num_predict_token
        self.max_seq_len = max_seq_len
        self.scaling = scaling
        self.quantile_levels = quantile_levels
        self.num_quantiles = len(quantile_levels)

        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.in_proj = ResidualBlock(
            input_dims=patch_size * 2,
            hidden_dims=d_model,
            output_dims=d_model,
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
        self.out_proj = ResidualBlock(
            input_dims=d_model,
            hidden_dims=d_model,
            output_dims=num_predict_token * self.num_quantiles * patch_size,
        )

    def forward(
        self,
        target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
        training_mode=True,
    ):
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        input_tokens = torch.cat(
            [scaled_target, observed_mask.to(torch.float32)], dim=-1
        )
        reprs = self.in_proj(input_tokens)

        reprs = self.encoder(
            reprs,
            packed_causal_attention_mask(sample_id, time_id),
            time_id=time_id,
            var_id=variate_id,
        )
        preds = self.out_proj(reprs)
        if training_mode:
            return preds, scaled_target
        else:
            return preds * scale + loc
