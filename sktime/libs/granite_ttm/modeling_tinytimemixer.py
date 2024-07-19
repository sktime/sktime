"""PyTorch TinyTimeMixer Model Implementation."""

import copy
import math
from dataclasses import dataclass
from typing import Optional
from warnings import warn

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.libs.granite_ttm.configuration_tinytimemixer import TinyTimeMixerConfig

if _check_soft_dependencies("transformers", severity="none"):
    from transformers.modeling_utils import ModelOutput, PreTrainedModel
else:

    class PreTrainedModel:
        """Dummy class if transformers is unavailable."""

    class ModelOutput:
        """Dummy class if transformers is unavailable."""


if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn

    nn_module = nn.Module
    torch_tensor = torch.Tensor
    torch_float = torch.FloatTensor
else:

    class nn_module:
        """Dummy class if torch is unavailable."""

    class torch_tensor:
        """Dummy class if torch is unavailable."""

    class torch_float:
        """Dummy class if torch is unavailable."""


class TinyTimeMixerGatedAttention(nn_module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        """Forward Pass."""
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


class TinyTimeMixerBatchNorm(nn_module):
    """
    TinyTimeMixerBatchNorm.

    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch_tensor):
        """
        Forward Pass.

        Args:
            inputs (`torch.Tensor` of
                shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Return:
            `torch.Tensor` of
                shape `(batch_size, sequence_length, d_model)`
        """
        output = inputs.transpose(1, 2)
        output = self.batchnorm(output)
        return output.transpose(1, 2)


class TinyTimeMixerPositionalEncoding(nn_module):
    """Class for positional encoding."""

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        if config.use_positional_encoding:
            self.position_enc = self._init_pe(config)
        else:
            self.position_enc = nn.Parameter(
                torch.zeros(config.num_patches, config.d_model)
            )

    @staticmethod
    def _init_pe(config: TinyTimeMixerConfig):
        if config.positional_encoding_type == "random":
            position_enc = nn.Parameter(
                torch.randn(config.num_patches, config.d_model), requires_grad=True
            )
        elif config.positional_encoding_type == "sincos":
            position_enc = torch.zeros(config.num_patches, config.d_model)
            position = torch.arange(0, config.num_patches).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, config.d_model, 2)
                * -(math.log(10000.0) / config.d_model)
            )
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            raise ValueError(
                f"{config.positional_encoding_type} is not a valid positional encoder. "
                "Available types are 'random' and 'sincos'."
            )
        return position_enc

    def forward(self, patch_input: torch_tensor):
        """Forward Pass."""
        hidden_state = patch_input + self.position_enc
        return hidden_state


class TinyTimeMixerNormLayer(nn_module):
    """Normalization block.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm_mlp = config.norm_mlp

        if "batch" in config.norm_mlp.lower():
            self.norm = TinyTimeMixerBatchNorm(config)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch_tensor):
        """
        Forward Pass.

        Args:
            inputs (`torch.Tensor` of
                shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Return:
            `torch.Tensor` of
                shape `((batch_size, num_channels, num_patches, d_model))`
        """
        if "batch" in self.norm_mlp.lower():
            inputs_reshaped = torch.reshape(
                inputs,
                (
                    inputs.shape[0] * inputs.shape[1],
                    inputs.shape[2],
                    inputs.shape[3],
                ),
            )

            inputs_reshaped = self.norm(inputs_reshaped)

            inputs = torch.reshape(inputs_reshaped, inputs.shape)

        else:
            inputs = self.norm(inputs)

        return inputs


class TinyTimeMixerMLP(nn_module):
    """TinyTimeMixerMLP."""

    def __init__(self, in_features, out_features, config):
        super().__init__()
        num_hidden = in_features * config.expansion_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, inputs: torch_tensor):
        """
        Forward Pass.

        Args:
            inputs (`torch.Tensor` of
                shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the MLP layer.
        Return:
            `torch.Tensor` of the same shape as `inputs`
        """
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class TinyTimeMixerChannelFeatureMixerBlock(nn_module):
    """
    TinyTimeMixerChannelFeatureMixerBlock.

    This module mixes the features in the channel dimension.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)
        self.gated_attn = config.gated_attn
        self.mlp = TinyTimeMixerMLP(
            in_features=config.num_input_channels,
            out_features=config.num_input_channels,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(
                in_size=config.num_input_channels, out_size=config.num_input_channels
            )

    def forward(self, inputs: torch_tensor):
        """
        Forward Pass.

        Args:
            inputs (`torch.Tensor` of
                shape `((batch_size, num_channels, num_patches, d_model))`):
                input to the MLP layer
        Return:
            `torch.Tensor` of the same shape as `inputs`
        """
        residual = inputs
        inputs = self.norm(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        if self.gated_attn:
            inputs = self.gating_block(inputs)

        inputs = self.mlp(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        out = inputs + residual
        return out


class TinyTimeMixerAttention(nn_module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[TinyTimeMixerConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch_tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch_tensor,
        key_value_states: Optional[torch_tensor] = None,
        past_key_value: Optional[tuple[torch_tensor]] = None,
        attention_mask: Optional[torch_tensor] = None,
        layer_head_mask: Optional[torch_tensor] = None,
        output_attentions: bool = False,
    ):
        """Input shape: Batch x Time x Channel."""
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling

        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                "Attention weights should be of size "
                f"{(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    "Head mask for a single layer should be of size "
                    f"{(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                "`attn_output` should be of size "
                f"{(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class PatchMixerBlock(nn_module):
    """
    PatchMixerBlock.

    This module mixes the patch dimension.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)

        self.self_attn = config.self_attn
        self.gated_attn = config.gated_attn

        self.mlp = TinyTimeMixerMLP(
            in_features=config.num_patches,
            out_features=config.num_patches,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(
                in_size=config.num_patches, out_size=config.num_patches
            )

        if config.self_attn:
            self.self_attn_layer = TinyTimeMixerAttention(
                embed_dim=config.d_model,
                num_heads=config.self_attn_heads,
                dropout=config.dropout,
            )
            self.norm_attn = TinyTimeMixerNormLayer(config)

    def forward(self, hidden_state):
        """
        Forward Pass.

        Args:
            hidden_state (`torch.Tensor`): Input tensor.

        Return:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden_state

        hidden_state = self.norm(hidden_state)

        if self.self_attn:
            batch_size, n_vars, num_patches, d_model = hidden_state.shape
            hidden_state_reshaped = hidden_state.reshape(
                batch_size * n_vars, num_patches, d_model
            )

            x_attn, _, _ = self.self_attn_layer(
                hidden_state_reshaped, output_attentions=False
            )
            x_attn = x_attn.reshape(batch_size, n_vars, num_patches, d_model)

        hidden_state = hidden_state.transpose(2, 3)
        hidden_state = self.mlp(hidden_state)

        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)

        hidden_state = hidden_state.transpose(2, 3)

        if self.self_attn:
            hidden_state = self.norm_attn(hidden_state + x_attn)

        out = hidden_state + residual
        return out


class FeatureMixerBlock(nn_module):
    """
    FeatureMixerBlock.

    This module mixes the hidden feature dimension.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)

        self.gated_attn = config.gated_attn

        self.mlp = TinyTimeMixerMLP(
            in_features=config.d_model,
            out_features=config.d_model,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(
                in_size=config.d_model, out_size=config.d_model
            )

    def forward(self, hidden: torch_tensor):
        """
        Forward Pass.

        Args:
            hidden (`torch.Tensor` of
                shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Return:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)

        if self.gated_attn:
            hidden = self.gating_block(hidden)

        out = hidden + residual
        return out


class TinyTimeMixerLayer(nn_module):
    """
    The `TinyTimeMixer` layer that does all three kinds of mixing.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        if config.num_patches > 1:
            self.patch_mixer = PatchMixerBlock(config=config)

        self.feature_mixer = FeatureMixerBlock(config=config)

        self.mode = config.mode
        self.num_patches = config.num_patches
        if config.mode == "mix_channel":
            self.channel_feature_mixer = TinyTimeMixerChannelFeatureMixerBlock(
                config=config
            )

    def forward(self, hidden: torch_tensor):
        """
        Forward Pass.

        Args:
            hidden (`torch.Tensor` of
                shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Return:
            `torch.Tensor`: Transformed tensor.
        """
        if self.mode == "mix_channel":
            hidden = self.channel_feature_mixer(hidden)

        if self.num_patches > 1:
            hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)
        return hidden


class TinyTimeMixerAdaptivePatchingBlock(nn_module):
    """
    The `TinyTimeMixer` layer that does all three kinds of mixing.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TinyTimeMixerConfig, adapt_patch_level: int):
        super().__init__()
        temp_config = copy.deepcopy(config)
        self.adapt_patch_level = adapt_patch_level
        adaptive_patch_factor = 2**adapt_patch_level
        self.adaptive_patch_factor = adaptive_patch_factor

        if config.d_model // self.adaptive_patch_factor <= 4:
            warn(
                f"Disabling adaptive patching at level {adapt_patch_level}. "
                "Either increase d_model or reduce adaptive_patching_levels"
            )
            self.adaptive_patch_factor = 1

        if config.d_model % self.adaptive_patch_factor != 0:
            raise ValueError(
                "d_model should be divisible by 2^i, where i varies "
                "from 0 to adaptive_patching_levels."
            )
        temp_config.num_patches = temp_config.num_patches * self.adaptive_patch_factor
        temp_config.d_model = temp_config.d_model // self.adaptive_patch_factor

        self.mixer_layers = nn.ModuleList(
            [TinyTimeMixerLayer(temp_config) for i in range(temp_config.num_layers)]
        )

    def forward(self, hidden: torch_tensor):
        """
        Forward Pass.

        Args:
            hidden (`torch.Tensor` of
                shape `(batch_size x nvars x num_patch x d_model)`):
                Input tensor to the layer.

        Return:
            `torch.Tensor`: Transformed tensor.
        """
        all_hidden_states = []
        all_hidden_states.append(hidden)
        hidden = torch.reshape(
            hidden,
            (
                hidden.shape[0],
                hidden.shape[1],
                hidden.shape[2] * self.adaptive_patch_factor,
                hidden.shape[3] // self.adaptive_patch_factor,
            ),
        )
        all_hidden_states.append(hidden)

        for mod in self.mixer_layers:
            hidden = mod(hidden)
            all_hidden_states.append(hidden)

        hidden = torch.reshape(
            hidden,
            (
                hidden.shape[0],
                hidden.shape[1],
                hidden.shape[2] // self.adaptive_patch_factor,
                hidden.shape[3] * self.adaptive_patch_factor,
            ),
        )
        all_hidden_states.append(hidden)

        return hidden, all_hidden_states


class TinyTimeMixerBlock(nn_module):
    """The main computing framework of the `TinyTimeMixer` model.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        num_layers = config.num_layers

        self.adaptive_patching_levels = config.adaptive_patching_levels

        if self.adaptive_patching_levels > 0:
            self.mixers = nn.ModuleList(
                [
                    TinyTimeMixerAdaptivePatchingBlock(
                        config=config, adapt_patch_level=i
                    )
                    for i in reversed(range(config.adaptive_patching_levels))
                ]
            )

        else:
            self.mixers = nn.ModuleList(
                [TinyTimeMixerLayer(config=config) for _ in range(num_layers)]
            )

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Forward Pass.

        Args:
            hidden_state (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Return:
            `torch.Tensor`: The embedding. `list`:
                List of all hidden states if `output_hidden_states` is set to `True`.
        """
        all_hidden_states = []

        embedding = hidden_state

        for mod in self.mixers:
            if self.adaptive_patching_levels > 0:
                embedding, hidden_states = mod(embedding)
                all_hidden_states.extend(hidden_states)
            else:
                embedding = mod(embedding)
                if output_hidden_states:
                    all_hidden_states.append(embedding)

        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None


class TinyTimeMixerDecoder(nn_module):
    """Decoder for tiny time mixer.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        if config.d_model != config.decoder_d_model:
            self.adapter = nn.Linear(config.d_model, config.decoder_d_model)
        else:
            self.adapter = None

        self.decoder_raw_residual = config.decoder_raw_residual
        self.num_input_channels = config.num_input_channels

        if config.decoder_raw_residual:
            self.decoder_raw_embedding = nn.Linear(
                config.patch_length, config.decoder_d_model
            )

        decoder_config = copy.deepcopy(config)
        decoder_config.num_layers = config.decoder_num_layers
        decoder_config.d_model = config.decoder_d_model
        decoder_config.dropout = config.head_dropout
        decoder_config.adaptive_patching_levels = (
            config.decoder_adaptive_patching_levels
        )
        decoder_config.mode = config.decoder_mode

        self.decoder_block = TinyTimeMixerBlock(decoder_config)

        self.resolution_prefix_tuning = config.resolution_prefix_tuning

    def forward(
        self,
        hidden_state,
        patch_input,
        output_hidden_states: bool = False,
    ):
        """
        Forward Pass.

        Args:
            hidden_state (`torch.Tensor` of
                shape `(batch_size x nvars x num_patch x d_model)`):
                The input tensor from backbone.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.


        Return:
            `torch.Tensor`: The embedding. `list`: List of all hidden
                states if `output_hidden_states` is set to `True`.
        """
        if output_hidden_states:
            decoder_hidden_states = []
        else:
            decoder_hidden_states = None

        decoder_input = hidden_state

        if self.adapter is not None:
            decoder_input = self.adapter(hidden_state)
            if output_hidden_states:
                decoder_hidden_states.append(decoder_input)

        if self.decoder_raw_residual:
            if self.resolution_prefix_tuning:
                if patch_input.shape[-2] == decoder_input.shape[-2] - 1:
                    temp_shape = list(patch_input.shape)
                    temp_shape[-2] = 1
                    temp_zeros = torch.zeros(*temp_shape).to(patch_input.device)
                    patch_input = torch.cat([temp_zeros, patch_input], dim=-2)

            decoder_input = decoder_input + self.decoder_raw_embedding(patch_input)
            if output_hidden_states:
                decoder_hidden_states.append(decoder_input)

        decoder_output, hidden_states = self.decoder_block(
            hidden_state=decoder_input, output_hidden_states=output_hidden_states
        )

        if output_hidden_states:
            decoder_hidden_states.extend(hidden_states)

        return decoder_output, decoder_hidden_states


class TinyTimeMixerForPredictionHead(nn_module):
    """Prediction Head for Forecasting.

    Args:
        config (`TinyTimeMixerConfig`, *required*): Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.prediction_channel_indices = config.prediction_channel_indices

        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices.sort()

        self.prediction_filter_length = config.prediction_filter_length

        self.dropout_layer = nn.Dropout(config.head_dropout)
        if config.use_decoder:
            head_d_model = config.decoder_d_model
        else:
            head_d_model = config.d_model

        self.base_forecast_block = nn.Linear(
            (config.num_patches * head_d_model), config.prediction_length
        )

        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, hidden_features, past_values, future_values=None):
        """
        Forward Pass.

        Args:
            hidden_features `(batch_size, n_vars, num_patch, d_model)`
            in `common_channel`/`mix_channel` mode.): Input hidden
                features.

            past_values (`torch.FloatTensor` of
                shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a forecasting task,
            this denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension
            should be 1. For multivariate time series, it is
            greater than 1.

            future_values (`torch.Tensor` of
                shape `(batch_size, prediction length, input_channels)`,
                *optional*, Defaults to None):
                Actual groundtruths of the forecasts. Pass dummy values
                (say 0) for forecast channels, if groundtruth is unknown.
                Pass the correct values for Exogenous channels
                where the forecast values are known.


        Return:
            `torch.Tensor` of
                shape `(batch_size, prediction_length, forecast_channels)`.

        """
        hidden_features = self.flatten(hidden_features)
        hidden_features = self.dropout_layer(hidden_features)
        forecast = self.base_forecast_block(hidden_features)
        if isinstance(forecast, tuple):
            forecast = tuple(z.transpose(-1, -2) for z in forecast)
        else:
            forecast = forecast.transpose(-1, -2)

        if self.prediction_channel_indices is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(
                    z[..., self.prediction_channel_indices] for z in forecast
                )
            else:
                forecast = forecast[..., self.prediction_channel_indices]

        if self.prediction_filter_length is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(
                    z[:, : self.prediction_filter_length, :] for z in forecast
                )
            else:
                forecast = forecast[:, : self.prediction_filter_length, :]

        if (
            self.prediction_filter_length is not None
            and future_values is not None
            and future_values.shape[1] != self.prediction_filter_length
        ):
            future_values = future_values[:, : self.prediction_filter_length, :]

        return forecast


class TinyTimeMixerPreTrainedModel(PreTrainedModel):
    """TinyTimeMixerPreTrainedModel."""

    config_class = TinyTimeMixerConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, TinyTimeMixerPositionalEncoding):
            if self.config.positional_encoding_type == "random":
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, TinyTimeMixerBatchNorm):
            module.batchnorm.bias.data.zero_()
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()


class TinyTimeMixerPatchify(nn_module):
    """
    A class to patchify the time series sequence into different patches.

    Return:
        `torch.Tensor` of
            shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.sequence_length = config.context_length
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater "
                f"than the patch length ({self.patch_length})"
            )

        self.num_patches = (
            max(self.sequence_length, self.patch_length) - self.patch_length
        ) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (
            self.num_patches - 1
        )
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, past_values: torch_tensor):
        """
        Forward Pass.

        Args:
            past_values (`torch.Tensor` of
                shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Return:
            `torch.Tensor` of
                shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        sequence_length = past_values.shape[-2]
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't "
                f"match model configuration ({self.sequence_length})."
            )

        output = past_values[:, self.sequence_start :, :]

        output = output.unfold(
            dimension=-2, size=self.patch_length, step=self.patch_stride
        )

        output = output.transpose(-2, -3).contiguous()
        return output


class TinyTimeMixerStdScaler(nn_module):
    """
    TinyTimeMixerStdScaler.

    Standardize features by calculating the mean and scaling along the
    first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = (
            config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5
        )

    def forward(self, data: torch_tensor, observed_indicator: torch_tensor):
        """
        Forward Pass.

        Args:
            data (`torch.Tensor` of
                shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of
                shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Return:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * observed_indicator).sum(
            self.dim, keepdim=self.keepdim
        ) / denominator

        variance = (((data - loc) * observed_indicator) ** 2).sum(
            self.dim, keepdim=self.keepdim
        ) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


class TinyTimeMixerMeanScaler(nn_module):
    """
    TinyTimeMixerMeanScaler.

    Computes a scaling factor as the weighted average absolute value
    along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = (
            config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        )
        self.default_scale = (
            config.default_scale if hasattr(config, "default_scale") else None
        )

    def forward(self, data: torch_tensor, observed_indicator: torch_tensor):
        """
        Forward Pass.

        Args:
            data (`torch.Tensor` of
                shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of
                shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Return:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        scale = torch.where(num_observed > 0, scale, default_scale)

        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


class TinyTimeMixerNOPScaler(nn_module):
    """
    TinyTimeMixerNOPScaler.

    Assigns a scaling factor equal to 1 along the first dimension,
    and therefore applies no scaling to the input data.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    def forward(self, data: torch_tensor, observed_indicator: torch_tensor = None):
        """
        Forward Pass.

        Args:
            data (`torch.Tensor` of
                shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
        Return:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        scale = torch.ones_like(data, requires_grad=False).mean(
            dim=self.dim, keepdim=self.keepdim
        )
        loc = torch.zeros_like(data, requires_grad=False).mean(
            dim=self.dim, keepdim=self.keepdim
        )
        return data, loc, scale


@dataclass
class TinyTimeMixerEncoderOutput(ModelOutput):
    """
    Base class for `TinyTimeMixerEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of
            shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch_float = None
    hidden_states: Optional[tuple[torch_float]] = None


class TinyTimeMixerEncoder(TinyTimeMixerPreTrainedModel):
    """
    TinyTimeMixerEncoder.

    Encoder for TinyTimeMixer which inputs patched
    time-series and outputs patched embeddings.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        if config.init_processing is False:
            config.check_and_init_preprocessing()

        super().__init__(config)

        self.use_return_dict = config.use_return_dict

        self.patcher = nn.Linear(config.patch_length, config.d_model)
        if config.use_positional_encoding:
            self.positional_encoder = TinyTimeMixerPositionalEncoding(config=config)
        else:
            self.positional_encoder = None
        self.mlp_mixer_encoder = TinyTimeMixerBlock(config=config)

        if config.resolution_prefix_tuning:
            mid_dim = (config.patch_length + config.d_model) // 2

            self.freq_mod = nn.Sequential(
                nn.Embedding(config.frequency_token_vocab_size, config.patch_length),
                nn.Linear(config.patch_length, mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, config.d_model),
            )
        self.resolution_prefix_tuning = config.resolution_prefix_tuning
        self.d_model = config.d_model

        if config.post_init:
            self.post_init()

    def forward(
        self,
        past_values: torch_tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch_tensor] = None,
    ):
        r"""
        Forward Pass.

        Args:
            past_values (`torch.FloatTensor` of
                shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series.
                For univariate time series, `num_input_channels` dimension should be 1.
                For multivariate time series,
                it is greater than 1.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`]
                instead of a plain tuple.

        Return:
            `torch.FloatTensor` of
                shape `(batch_size, n_vars, num_patches, d_model)`
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        patches = self.patcher(past_values)

        if self.resolution_prefix_tuning:
            if freq_token is not None:
                freq_embedding = self.freq_mod(freq_token.long())

                freq_embedding = freq_embedding.view(
                    patches.shape[0], 1, 1, self.d_model
                )
                freq_embedding = freq_embedding.expand(
                    patches.shape[0],
                    patches.shape[1],
                    1,
                    self.d_model,
                )

                patches = torch.cat((freq_embedding, patches), dim=-2)

            else:
                raise Exception("Expecting freq_token in forward")

        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)

        last_hidden_state, hidden_states = self.mlp_mixer_encoder(
            patches, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    hidden_states,
                ]
            )

        return TinyTimeMixerEncoderOutput(
            last_hidden_state=last_hidden_state, hidden_states=hidden_states
        )


@dataclass
class TinyTimeMixerModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`  of
        shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        patch_input (`torch.FloatTensor` of shape
            `(batch_size, num_channels, num_patches, patch_length)`):
            Patched input data to the model.
        loc: (`torch.FloatTensor` of
            shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the mean of the context window per channel.
            Used for revin denorm outside the model, if revin
            enabled.
        scale: (`torch.FloatTensor` of
            shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the std dev of the context window per channel.
            Used for revin denorm outside the model, if revin
            enabled.
    """

    last_hidden_state: torch_float = None
    hidden_states: Optional[tuple[torch_float]] = None
    patch_input: torch_float = None
    loc: Optional[torch_float] = None
    scale: Optional[torch_float] = None


class TinyTimeMixerModel(TinyTimeMixerPreTrainedModel):
    """TinyTimeMixerModel.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for
    the generic methods the
    library implements for all its model (such as downloading or saving,
    resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch
    [nn_module](https://pytorch.org/docs/stable/nn.html#torch.nn_module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation
    for all matter related to general usage
    and behavior.

    Args:
        config ([`TinyTimeMixerConfig`]):
            Model configuration class with all the parameters of the model. Initializing
            with a config file does not
            load the weights associated with the model, only the configuration.
            Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        if config.init_processing is False:
            config.check_and_init_preprocessing()

        super().__init__(config)

        self.use_return_dict = config.use_return_dict
        self.encoder = TinyTimeMixerEncoder(config)
        self.patching = TinyTimeMixerPatchify(config)

        if config.scaling == "mean":
            self.scaler = TinyTimeMixerMeanScaler(config)
        elif config.scaling == "std" or config.scaling is True:
            self.scaler = TinyTimeMixerStdScaler(config)
        else:
            self.scaler = TinyTimeMixerNOPScaler(config)

        self.d_model = config.d_model

        if config.post_init:
            self.post_init()

    def forward(
        self,
        past_values: torch_tensor,
        observed_mask: Optional[torch_tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch_tensor] = None,
    ):
        r"""
        Forward Pass.

        Args:
            past_values (`torch.FloatTensor` of
                shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series. For a forecasting task,
                this denotes the
                history/past time series values.
                For univariate time series, `num_input_channels` dimension should be 1.
                For multivariate time series, it is
                greater than 1.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a
                plain tuple.

            observed_mask (`torch.FloatTensor` of shape
                `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values`
                were observed and which were missing. Mask values selected
                in `[0, 1]`:
                    - 1 for values that are **observed**,
                    - 0 for values that are **missing**
                    (i.e. NaNs that were replaced by zeros).

        Returns
        -------
            `TinyTimeMixerModelOutput` or `tuple`:
                If `return_dict` is True, returns a `TinyTimeMixerModelOutput` object,
                otherwise returns a tuple.
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if observed_mask is None:
            observed_mask = torch.ones_like(past_values)
        scaled_past_values, loc, scale = self.scaler(past_values, observed_mask)

        patched_x = self.patching(scaled_past_values)

        enc_input = patched_x

        encoder_output = self.encoder(
            enc_input,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            freq_token=freq_token,
        )

        if isinstance(encoder_output, tuple):
            encoder_output = TinyTimeMixerEncoderOutput(*encoder_output)

        if not return_dict:
            return tuple(
                v
                for v in [
                    encoder_output.last_hidden_state,
                    encoder_output.hidden_states,
                    patched_x,
                    loc,
                    scale,
                ]
            )

        return TinyTimeMixerModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            patch_input=patched_x,
            loc=loc,
            scale=scale,
        )


@dataclass
class TinyTimeMixerForPredictionOutput(ModelOutput):
    """
    Output type of [`TinyTimeMixerForPredictionOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape
        `(batch_size, prediction_length, num_input_channels)`):
            Prediction output from the forecast head.
        backbone_hidden_state (`torch.FloatTensor` of shape
        `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the decoder
        decoder_hidden_state (`torch.FloatTensor` of shape
        `(batch_size, num_input_channels, num_patches, d_model)`):
            Decoder embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer
            plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of
            shape `()`):
            Total loss.
        loc (`torch.FloatTensor`, *optional* of shape
        `(batch_size, 1, num_input_channels)`):
            Input mean
        scale (`torch.FloatTensor`, *optional* of shape
        `(batch_size, 1, num_input_channels)`):
            Input std dev

    """

    loss: Optional[torch_float] = None
    prediction_outputs: torch_float = None
    backbone_hidden_state: torch_float = None
    decoder_hidden_state: torch_float = None
    hidden_states: Optional[tuple[torch_float]] = None
    loc: torch_float = None
    scale: torch_float = None


class TinyTimeMixerForPrediction(TinyTimeMixerPreTrainedModel):
    r"""
    `TinyTimeMixer` for forecasting application.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    Return:
        `None`.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        config.check_and_init_preprocessing()

        super().__init__(config)

        self.loss = config.loss

        self.use_return_dict = config.use_return_dict

        self.prediction_channel_indices = config.prediction_channel_indices

        self.num_input_channels = config.num_input_channels

        self.prediction_filter_length = config.prediction_filter_length

        self.backbone = TinyTimeMixerModel(config)

        self.use_decoder = config.use_decoder

        if config.use_decoder:
            self.decoder = TinyTimeMixerDecoder(config)

        self.head = TinyTimeMixerForPredictionHead(
            config=config,
        )

        if config.post_init:
            self.post_init()

    def forward(
        self,
        past_values: torch_tensor,
        future_values: Optional[torch_tensor] = None,
        observed_mask: Optional[torch_tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch_tensor] = None,
    ):
        r"""
        Forward Pass.

        Args:
            past_values (`torch.FloatTensor` of
                shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series. For a forecasting task,
                this denotes the
                history/past time series values.
                For univariate time series, `num_input_channels` dimension should be 1.
                For multivariate time series, it is
                greater than 1.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of
                a plain tuple.
            observed_mask (`torch.FloatTensor` of shape
                `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed
                and which were missing. Mask values selected
                in `[0, 1]`:
                    - 1 for values that are **observed**,
                    - 0 for values that are **missing**

                    y zeros).
            future_values (`torch.FloatTensor` of shape
                `(batch_size, target_len, num_input_channels)` for forecasting,:
                `(batch_size, num_targets)` for regression, or `(batch_size,)`
                for classification, *optional*): Target
                values of the time series, that serve as labels for the model.
                The `future_values` is what the
                Transformer needs during training to learn to output, given the
                `past_values`. Note that, this is NOT
                required for a pretraining task.

                For a forecasting task, the shape is be
                `(batch_size, target_len, num_input_channels)`. Even if we want
                to forecast only specific channels by setting the indices in
                `prediction_channel_indices` parameter,
                pass the target data with all channels, as channel Filtering for both
                prediction and target will be
                manually applied before the loss computation.
            return_loss (`bool`,  *optional*):
                Whether to return the loss in the `forward` call.

        Return:

        """
        if self.loss == "mse":
            loss = nn.MSELoss(reduction="mean")
        elif self.loss == "mae":
            loss = nn.L1Loss(reduction="mean")
        else:
            loss = None

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        model_output = self.backbone(
            past_values,
            observed_mask=observed_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            freq_token=freq_token,
        )

        if isinstance(model_output, tuple):
            model_output = TinyTimeMixerModelOutput(*model_output)

        decoder_input = model_output.last_hidden_state
        hidden_states = model_output.hidden_states

        if self.use_decoder:
            decoder_output, decoder_hidden_states = self.decoder(
                hidden_state=decoder_input,
                patch_input=model_output.patch_input,
                output_hidden_states=output_hidden_states,
            )

            if decoder_hidden_states:
                hidden_states.extend(decoder_hidden_states)

        else:
            decoder_output = decoder_input

        y_hat = self.head(
            decoder_output, past_values=past_values, future_values=future_values
        )

        if (
            self.prediction_filter_length is not None
            and future_values is not None
            and future_values.shape[1] != self.prediction_filter_length
        ):
            future_values = future_values[:, : self.prediction_filter_length, :]

        if (
            self.prediction_channel_indices is not None
            and future_values is not None
            and future_values.shape[2] != len(self.prediction_channel_indices)
            and future_values.shape[2] == self.num_input_channels
        ):
            future_values = future_values[..., self.prediction_channel_indices]

        if self.prediction_channel_indices is not None:
            loc = model_output.loc[..., self.prediction_channel_indices]
            scale = model_output.scale[..., self.prediction_channel_indices]
        else:
            loc = model_output.loc
            scale = model_output.scale

        loss_val = None

        y_hat = y_hat * scale + loc

        if future_values is not None and return_loss is True and loss is not None:
            loss_val = loss(y_hat, future_values)

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    y_hat,
                    model_output.last_hidden_state,
                    decoder_output,
                    hidden_states,
                    loc,
                    scale,
                ]
            )

        return TinyTimeMixerForPredictionOutput(
            loss=loss_val,
            prediction_outputs=y_hat,
            backbone_hidden_state=model_output.last_hidden_state,
            decoder_hidden_state=decoder_output,
            hidden_states=hidden_states,
            loc=loc,
            scale=scale,
        )
