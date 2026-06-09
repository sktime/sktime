# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http:www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch model definitions for TimerS1 forecasting models."""

import math
from dataclasses import dataclass

from sktime.utils.dependencies import _check_soft_dependencies

from .configuration_TimerS1 import TimerS1Config
from .ts_generation_mixin import TSGenerationMixin

if _check_soft_dependencies("torch", "transformers", severity="none"):
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import Cache, DynamicCache, PreTrainedModel
    from transformers.activations import ACT2FN
    from transformers.modeling_attn_mask_utils import (
        _prepare_4d_causal_attention_mask,
    )
    from transformers.modeling_outputs import (
        MoeCausalLMOutputWithPast,
        MoeModelOutputWithPast,
    )
else:

    class _TimerS1DependencyStub:
        """Placeholder base used when TimerS1 soft dependencies are unavailable."""

    class _TimerS1NNStub:
        """Minimal ``torch.nn`` placeholder for import-time class definitions."""

        Module = object

    class _TimerS1TorchStub:
        """Minimal ``torch`` placeholder for import-time annotations."""

        Tensor = object
        FloatTensor = object
        LongTensor = object
        nn = _TimerS1NNStub

    def _prepare_4d_causal_attention_mask(*args, **kwargs):
        raise ModuleNotFoundError(
            "TimerS1 model execution requires the optional dependencies "
            "'torch' and 'transformers'."
        )

    torch = _TimerS1TorchStub()
    F = None
    nn = _TimerS1NNStub
    Cache = _TimerS1DependencyStub
    DynamicCache = _TimerS1DependencyStub
    PreTrainedModel = _TimerS1DependencyStub
    ACT2FN = {}
    MoeCausalLMOutputWithPast = _TimerS1DependencyStub
    MoeModelOutputWithPast = _TimerS1DependencyStub


@dataclass
class TimerS1CausalLMOutput(MoeCausalLMOutputWithPast):
    """Causal output with hidden states used by MTP generation.

    This registers hidden_states_for_mtp in the ModelOutput OrderedDict and
    makes it available through attribute access.
    """

    hidden_states_for_mtp: torch.FloatTensor | None = None


def _get_usable_past_kv_length(
    cache: Cache, new_seq_length: int, layer_idx: int = 0
) -> int:
    """Compute usable past length for a cache and new sequence length.

    This mirrors the previous `get_usable_length(new_seq_length, layer_idx)`
    behavior from Transformers < 4.45 while supporting the new Cache API.
    """
    try:
        previous_length = cache.get_seq_length(layer_idx)
        # Dynamic layers return -1, static layers return an int
        max_length = cache.get_max_cache_shape(layer_idx)
        if (
            max_length is not None
            and max_length != -1
            and previous_length + new_seq_length > max_length
        ):
            return max_length - new_seq_length
        return previous_length
    except Exception:
        # Best-effort fallback
        return (
            cache.get_seq_length(layer_idx) if hasattr(cache, "get_seq_length") else 0
        )


@dataclass
class TempMoeModelOutputWithPast(MoeModelOutputWithPast):
    """Moe model output carrying cache metadata used by TimerS1."""

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Cache | tuple[tuple[torch.Tensor, torch.Tensor]] | None = None
    use_legacy_cache: bool | None = None
    past_key_values_length: int | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_logits: tuple[torch.FloatTensor] | None = None


def rotate_half(x):
    """Rotate the last tensor dimension by half for rotary embeddings."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Apply rotary positional embeddings to query and key states."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """Root mean square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor along the last dimension."""
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return x_norm * self.weight


class ResidualBlock(nn.Module):
    """Residual projection block for TimerS1 prediction heads."""

    def __init__(self, config: TimerS1Config) -> None:
        super().__init__()
        self.out_dim = len(config.quantiles) * config.output_token_lens[-1]
        self.dropout = nn.Dropout(config.dropout_rate)
        self.hidden_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]
        self.output_layer = nn.Linear(config.hidden_size, self.out_dim)
        self.residual_layer = nn.Linear(config.hidden_size, self.out_dim)

    def forward(self, x: torch.Tensor):
        """Project hidden states to output patch values."""
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        return out + self.residual_layer(x)


class TimerS1PatchEmbedding(nn.Module):
    """Patch and embed continuous time-series inputs."""

    def __init__(self, config: TimerS1Config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.hidden_layer = nn.Linear(
            config.input_token_len * 2, config.intermediate_size
        )
        self.act = ACT2FN[config.hidden_act]
        self.output_layer = nn.Linear(config.intermediate_size, config.hidden_size)
        self.residual_layer = nn.Linear(config.input_token_len * 2, config.hidden_size)
        self.input_token_len = config.input_token_len

    def forward(self, x):
        """Convert a time-series tensor into patch embeddings."""
        mask = torch.ones_like(x)
        input_length = x.shape[-1]
        padding_length = (
            self.input_token_len - (input_length % self.input_token_len)
        ) % self.input_token_len
        x = F.pad(x, (padding_length, 0))
        mask = F.pad(mask, (padding_length, 0))
        x = x.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)
        mask = mask.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len
        )
        x = torch.cat([x, mask], dim=-1)
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        return out + self.residual_layer(x)


class TimerS1RotaryEmbedding(torch.nn.Module):
    """Rotary embedding cache for TimerS1 attention layers."""

    def __init__(self, dim, max_position_embeddings=10000, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """Return cached cosine and sine embeddings for a sequence length."""
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class TimerS1Attention(nn.Module):
    """Multi-head self-attention with rotary embeddings and QK normalization."""

    def __init__(self, config: TimerS1Config, layer_idx: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.dropout_rate

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # QK-Norm learnable scales
        self.q_scale = nn.Parameter(torch.ones(self.head_dim))
        self.k_scale = nn.Parameter(torch.ones(self.head_dim))

        # Attention output gate
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.rotary_emb = TimerS1RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-6
        q = (
            q
            * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
            * self.q_scale.view(1, 1, 1, -1)
        )
        k = (
            k
            * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)
            * self.k_scale.view(1, 1, 1, -1)
        )
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Run self-attention for one decoder layer."""
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += _get_usable_past_kv_length(
                past_key_value, kv_seq_len, self.layer_idx
            )
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        query_states, key_states = self._apply_qk_norm(query_states, key_states)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout_p=(self.attention_dropout if self.training else 0.0),
        )  # [bsz, num_heads, q_len, head_dim]

        gate = torch.sigmoid(self.gate_proj(hidden_states))
        gate = gate.view(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_output = attn_output * gate

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, q_len, self.hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        attn_weights = None if not output_attentions else attn_output
        return attn_output, attn_weights, past_key_value


class TimerS1MLP(nn.Module):
    """Feed-forward block used by TimerS1 experts."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        """Apply gated feed-forward projection to hidden states."""
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


class TimerS1ExpertsLayer(nn.Module):
    """Sparse mixture-of-experts feed-forward layer."""

    def __init__(self, config: TimerS1Config):
        super().__init__()
        self.top_k = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        moe_intermediate_size = config.intermediate_size // self.top_k

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                TimerS1MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=moe_intermediate_size,
                    hidden_act=config.hidden_act,
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor):
        """Route hidden states through top-k experts."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(
            2, 1, 0
        )

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


class TimerS1DecoderLayer(nn.Module):
    """Decoder layer combining attention and mixture-of-experts blocks."""

    def __init__(self, config: TimerS1Config, layer_idx: int):
        super().__init__()
        self.self_attn = TimerS1Attention(config, layer_idx)
        self.ffn_layer = TimerS1ExpertsLayer(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs,
    ) -> tuple[torch.FloatTensor, torch.Tensor | None, Cache | None]:
        """Run one TimerS1 decoder layer."""
        residual = hidden_states
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=self.norm1(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_layer(self.norm2(hidden_states))
        hidden_states = residual + hidden_states

        if not output_attentions:
            self_attn_weights = None
        if not use_cache:
            present_key_value = None

        return hidden_states, self_attn_weights, present_key_value


class TimerS1PreTrainedModel(PreTrainedModel):
    """Base pretrained model class for TimerS1."""

    config_class = TimerS1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TimerS1DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class TimerS1Model(TimerS1PreTrainedModel):
    """TimerS1 decoder backbone."""

    def __init__(self, config: TimerS1Config):
        super().__init__(config)
        self.embed_layer = TimerS1PatchEmbedding(config)
        self.layers = nn.ModuleList(
            [
                TimerS1DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | tuple[tuple[torch.Tensor, torch.Tensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | MoeModelOutputWithPast:
        """Run the TimerS1 decoder backbone."""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_layer(input_ids)
            seq_length = inputs_embeds.shape[1]

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        past_key_values_length = 0
        use_legacy_cache = None
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = _get_usable_past_kv_length(
                past_key_values, seq_length
            )

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            ).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=None,
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_moe_losses = []

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attns,
                    all_moe_losses,
                ]
                if v is not None
            )

        return TempMoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            use_legacy_cache=use_legacy_cache,
            past_key_values_length=past_key_values_length,
            router_logits=all_moe_losses,
        )


class TimerS1MTPLayer(nn.Module):
    """Multi-token prediction layer for TimerS1 generation."""

    def __init__(self, config: TimerS1Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        self.norm_hidden = RMSNorm(config.hidden_size)
        self.norm_embeds = RMSNorm(config.hidden_size)
        self.projection_matrix = nn.Linear(
            2 * self.hidden_size, self.hidden_size, bias=False
        )
        self.layer = TimerS1DecoderLayer(
            config, self.layer_idx + self.config.num_hidden_layers
        )
        self.norm = RMSNorm(config.hidden_size)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | tuple[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_legacy_cache: bool | None = False,
        past_key_values_length: int | None = 0,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | MoeModelOutputWithPast:
        """Run one multi-token prediction layer."""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=None,
        )

        hidden_states = self.norm_hidden(hidden_states)
        inputs_embeds = self.norm_embeds(inputs_embeds)
        hidden_states = self.projection_matrix(
            torch.cat([hidden_states, inputs_embeds], dim=-1)
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_moe_losses = []
        next_decoder_cache = None

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                self.layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = self.layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if use_cache:
            next_decoder_cache = layer_outputs[2]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_moe_losses,
                ]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_moe_losses,
        )


class TimerS1ForPrediction(TimerS1PreTrainedModel, TSGenerationMixin):
    """TimerS1 model with prediction heads for time-series forecasting."""

    def __init__(self, config: TimerS1Config):
        super().__init__(config)
        self.config = config
        self.model = TimerS1Model(self.config)
        self.output_patch_embedding = ResidualBlock(config)
        self.num_quantiles = len(config.quantiles)
        if self.config.num_mtp_tokens > 0:
            self.mtp_modules = nn.ModuleList(
                [
                    TimerS1MTPLayer(config, layer_idx)
                    for layer_idx in range(self.config.num_mtp_tokens)
                ]
            )
        self.post_init()

    def set_decoder(self, decoder):
        """Set the decoder backbone."""
        self.model = decoder

    def get_decoder(self):
        """Return the decoder backbone."""
        return self.model

    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | tuple[tuple[torch.Tensor, torch.Tensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        full_input_ids: torch.FloatTensor | None = None,
        full_hidden_states: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        max_output_length: int | None = None,
        revin: bool | None = False,
    ) -> tuple | TimerS1CausalLMOutput:
        """Run TimerS1 forecasting prediction."""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if revin:
            means = input_ids.mean(1, keepdim=True).detach()
            stdev = input_ids.std(dim=1, keepdim=True, unbiased=False).detach()
            stdev = torch.where(
                stdev > 1e-2, stdev, torch.tensor(1.0, device=input_ids.device)
            )
            input_ids = (input_ids - means) / stdev
            if full_input_ids is not None:
                fi_means = full_input_ids.mean(1, keepdim=True).detach()
                fi_stdev = full_input_ids.std(
                    dim=1, keepdim=True, unbiased=False
                ).detach()
                fi_stdev = torch.where(
                    fi_stdev > 1e-2,
                    fi_stdev,
                    torch.tensor(1.0, device=full_input_ids.device),
                )
                full_input_ids = (full_input_ids - fi_means) / fi_stdev
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_layer(input_ids)
        # Embeddings for the complete sequence used by MTP layers without KV cache.
        if full_input_ids is not None:
            full_inputs_embeds = self.model.embed_layer(full_input_ids)
        else:
            full_inputs_embeds = inputs_embeds

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state

        # Accumulate full hidden states across generation steps for MTP layers.
        # When KV cache is enabled, hidden_states only covers new tokens, so prepend
        # accumulated past hidden states to restore the full MTP sequence picture.
        # When KV cache is disabled, hidden_states already covers the full sequence
        # (same length as full_inputs_embeds), so no accumulation is needed.
        if (
            full_hidden_states is not None
            and hidden_states.shape[1] < full_inputs_embeds.shape[1]
        ):
            mtp_hidden_states = torch.cat(
                [full_hidden_states.to(hidden_states.device), hidden_states], dim=1
            )
        else:
            mtp_hidden_states = hidden_states

        bsz, L, _ = hidden_states.shape
        predictions = None
        loss = None
        if max_output_length is None:
            output_token_len = self.config.output_token_lens[0]
            max_output_length = output_token_len
        else:
            output_token_len = self.config.output_token_lens[0]
            for h in self.config.output_token_lens[1:]:
                if h > max_output_length:
                    break
                output_token_len = h

        predictions = self.output_patch_embedding(hidden_states[:, -1, :]).reshape(
            bsz, self.num_quantiles, self.config.output_token_lens[-1]
        )

        if self.config.num_mtp_tokens > 0:
            output_patch_len = self.config.output_token_lens[-1]
            full_out_len = (
                output_patch_len
                + self.config.input_token_len * self.config.num_mtp_tokens
            )

            target_len = max(0, min(int(max_output_length), int(full_out_len)))

            out = torch.zeros(
                bsz, self.num_quantiles, target_len, device=predictions.device
            )
            base_fill = min(output_patch_len, target_len)
            if base_fill > 0:
                out[:, :, :base_fill] = predictions[:, :, :base_fill]

            if target_len <= output_patch_len:
                mtp_steps_needed = 0
            else:
                remaining = target_len - output_patch_len
                mtp_steps_needed = min(
                    self.config.num_mtp_tokens,
                    math.ceil(remaining / self.config.input_token_len),
                )

            for k, mtp_module in enumerate(self.mtp_modules):
                if k >= mtp_steps_needed:
                    break

                start_pos = (k + 1) * self.config.input_token_len
                if start_pos >= target_len:
                    break

                mtp_full_len = full_inputs_embeds.shape[1]
                mtp_attention_mask = (
                    attention_mask[:, -mtp_full_len:]
                    if attention_mask is not None
                    else None
                )
                mtp_outputs = mtp_module(
                    hidden_states=mtp_hidden_states,
                    inputs_embeds=full_inputs_embeds,
                    attention_mask=mtp_attention_mask,
                    output_attentions=output_attentions,
                )
                mtp_hidden_states = mtp_outputs[0]

                mtp_pred = self.output_patch_embedding(mtp_hidden_states)[:, -1, :]
                mtp_pred = mtp_pred.reshape(bsz, self.num_quantiles, output_patch_len)

                end_pos = min(start_pos + output_patch_len, target_len)
                take = end_pos - start_pos
                if take > 0:
                    out[:, :, start_pos:end_pos] = mtp_pred[:, :, :take]

            predictions = out

        if max_output_length is not None and predictions.shape[-1] > max_output_length:
            predictions = predictions[:, :, :max_output_length]
        if revin:
            predictions = predictions * stdev + means
        if not return_dict:
            output = (predictions,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TimerS1CausalLMOutput(
            loss=loss,
            logits=predictions,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            # Pass main-model hidden states as a proper field so that
            # _update_model_kwargs_for_generation can reliably accumulate them
            # for the MTP layers across multi-step generation.
            hidden_states_for_mtp=hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        revin=False,
        **kwargs,
    ):
        """Prepare TimerS1 inputs for autoregressive generation."""
        # full_input_ids always holds the complete original sequence for MTP layers
        full_input_ids = input_ids.clone()
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length(0)
                past_length = cache_length
                try:
                    max_cache_length = past_key_values.get_max_cache_shape(0)
                    if max_cache_length == -1:
                        max_cache_length = None
                except Exception:
                    max_cache_length = None
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Trim input_ids to only include unprocessed tokens
            if attention_mask is not None and attention_mask.shape[1] > (
                input_ids.shape[1] // self.config.input_token_len
            ):
                input_ids = input_ids[
                    :,
                    -(attention_mask.shape[1] - past_length)
                    * self.config.input_token_len :,
                ]
            elif past_length < (input_ids.shape[1] // self.config.input_token_len):
                input_ids = input_ids[:, past_length * self.config.input_token_len :]

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + (input_ids.shape[1] // self.config.input_token_len)
                > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_length > 0:
                position_ids = position_ids[
                    :, -(input_ids.shape[1] // self.config.input_token_len) :
                ]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "revin": revin,
                "full_input_ids": full_input_ids,
                "full_hidden_states": kwargs.get("full_hidden_states"),
            }
        )
        return model_inputs
