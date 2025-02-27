# Copyright 2023, Salesforce, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TimeMOE model implementation."""

import math
import warnings
from typing import Optional, Union

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:

    class torch:
        """Dummy class if torch is not installed."""

        class Tensor:
            """Dummy class if torch is not installed."""

        class LongTensor:
            """Dummy class if torch is not installed."""

        class FloatTensor:
            """Dummy class if torch is not installed."""

    class nn:
        """Dummy class if torch is not installed."""

        class Module:
            """Dummy class if torch is not installed."""

    class F:
        """Dummy class if torch is not installed."""


if _check_soft_dependencies("transformers", severity="none"):
    from transformers import Cache, DynamicCache, PreTrainedModel, StaticCache
    from transformers.activations import ACT2FN
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    from transformers.modeling_outputs import (
        MoeCausalLMOutputWithPast,
        MoeModelOutputWithPast,
    )
else:

    class Cache:
        """Dummy class if transformers is not installed."""

    class StaticCache:
        """Dummy class if transformers is not installed."""

    class PreTrainedModel:
        """Dummy class if transformers is not installed."""

    class MoeModelOutputWithPast:
        """Dummy class if transformers is not installed."""

    class MoeCausalLMOutputWithPast:
        """Dummy class if transformers is not installed."""


from .timemoe_config import TimeMoeConfig
from .ts_generation_mixin import TSGenerationMixin


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], list[torch.Tensor]],
    top_k: int,
    num_experts: int = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    Compute auxiliary load balancing loss as in Switch Transformer.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
    This function implements the loss function presented in equations (4) - (6) of the
    paper. It aims at penalizing cases where the routing between experts is too
    unbalanced.

    Parameters
    ----------
    gate_logits (Union[`torch.Tensor`, tuple[torch.Tensor], list[torch.Tensor]):
        Logits from the `gate`, should be a tuple of model.config.num_hidden_layers
        tensors of shape [batch_size X sequence_length, num_experts].
    top_k (`int`)
        Selected Top k over the experts.
    attention_mask (`torch.Tensor`, None):
        The attention_mask used in forward function
        shape [batch_size X sequence_length] if not None.
    num_experts (`int`, *optional*):
        Number of experts

    Returns
    -------
        The auxiliary loss.
    """
    if (
        gate_logits is None
        or not isinstance(gate_logits, (tuple, list))
        or gate_logits[0] is None
    ):  # noqa: E501
        return 0.0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
    )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )  # noqa: E501

        # Compute mask which masks the padding tokens as 0 with the shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, 2, num_experts))
            .reshape(-1, 2, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)  # noqa: E501

        # Compute the mask that masks all padding tokens as 0 with the
        # same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(  # noqa: E501
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(
        tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0)
    )

    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Apply Rotary Position Embedding to the query and key tensors.

    Parameters
    ----------
    q (`torch.Tensor`): The query tensor.
    k (`torch.Tensor`): The key tensor.
    cos (`torch.Tensor`): The cosine part of the rotary embedding.
    sin (`torch.Tensor`): The sine part of the rotary embedding.
    position_ids (`torch.Tensor`):
        The position indices of the tokens corresponding to the query and key tensors. For example, this can be
        used to pass offsetted position ids when working with a KV-cache.
    unsqueeze_dim (`int`, *optional*, defaults to 1):
        The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze
        cos[position_ids] and sin[position_ids] so that they can be properly broadcasted
        to the dimensions of q and k. For example, note that cos[position_ids] and
        sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then,
        if q and k have the shape [batch_size, heads, seq_len, head_dim],
        then setting unsqueeze_dim=1 makes cos[position_ids] and sin[position_ids]
        broadcastable to the shapes of q and k. Similarly, if q and k have the shape
        [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

    Returns
    -------
    `tuple(torch.Tensor)`:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """  # noqa: E501
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimeMoeInputEmbedding(nn.Module):
    """Use a mlp layer for embedding the time-series."""

    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size  # default 1
        self.hidden_size = config.hidden_size
        self.emb_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.gate_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        Perform a forward pass through the model, to embed input data.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the activation function and embedding layer.
        """
        emb = self.act_fn(self.gate_layer(x)) * self.emb_layer(x)
        return emb


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with
# Mistral->TimeMOE
class TimeMoeRotaryEmbedding(nn.Module):
    """
    Implements the Rotary Position Embedding for TimeMOE.

    Parameters
    ----------
    dim : int
        The dimension of the embeddings.
    max_position_embeddings : int, optional
        The maximum number of position embeddings. Default is 2048.
    base : int, optional
        The base value for computing inverse frequency. Default is 10000.
    device : torch.device, optional
        The device on which to place the tensors. Default is None.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
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
        )  # noqa: E501
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),  # noqa: E501
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)  # noqa: E501

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation  # noqa: E501
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """
        Forward pass for the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, num_attention_heads, seq_len, head_size].
        seq_len : int, optional
            Sequence length of the input tensor. If provided and greater than
            `self.max_seq_len_cached`, the cosine and sine caches will be updated.

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing:

            - cos_cached: Cached cosine values up to the sequence length,
            converted to the input tensor's dtype.
            - sin_cached: Cached sine values up to the sequence length,
            converted to the input tensor's dtype.
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->TimeMOE
class TimeMoeRMSNorm(nn.Module):
    """
    Implement RMSNorm (Root Mean Square Layer Normalization) for TimeMOE as outlined in
    `transformers.models.llama.modeling_llama.LlamaRMSNorm`.

    Parameters
    ----------
    hidden_size : int
        The size of the hidden layer.
    eps : float, optional
        A small value to avoid division by zero. Default is 1e-6.
    """  # noqa: D205

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Perform a forward pass through the layer.

        Parameter
        ---------
        hidden_states: torch.Tensor
            The input tensor containing hidden states.

        Returns
        -------
            torch.Tensor: The output tensor after applying normalization and scaling.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TimeMoeTemporalBlock(nn.Module):
    """
    Temporal block for TimeMOE model implementing a gated feed-forward network.

    Parameters
    ----------
    hidden_size : int
        Size of the hidden layer
    intermediate_size : int
        Size of the intermediate layer
    hidden_act : str
        Name of the activation function
    """

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        """
        Forward pass through the temporal block.

        Parameters
        ----------
        hidden_state : torch.Tensor
            Input tensor of hidden states

        Returns
        -------
        torch.Tensor
            Transformed hidden states
        """
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


class TimeMoeMLP(TimeMoeTemporalBlock):
    """
    Multi-layer perceptron variant of TimeMoeTemporalBlock.

    Extends TimeMoeTemporalBlock to return None as second output for compatibility
    with sparse expert layers.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__(hidden_size, intermediate_size, hidden_act)

    def forward(self, hidden_state):
        """
        Forward pass through the MLP layer.

        Parameters
        ----------
        hidden_state : torch.Tensor
            Input tensor of hidden states

        Returns
        -------
        tuple[torch.Tensor, None]
            Transformed hidden states and None as router logits
        """
        return super().forward(hidden_state), None


class TimeMoeSparseExpertsLayer(nn.Module):
    """
    Sparse mixture of experts layer for TimeMOE model.

    This layer implements a sparse gating mechanism where each token is processed by a
    subset of experts, with routing determined by a learned gate network.

    Parameters
    ----------
    config : TimeMoeConfig
        Configuration object containing model hyperparameters.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.norm_topk_prob = False

        moe_intermediate_size = self.config.intermediate_size // self.top_k

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                TimeMoeTemporalBlock(
                    hidden_size=self.config.hidden_size,
                    intermediate_size=moe_intermediate_size,
                    hidden_act=self.config.hidden_act,
                )
                for _ in range(self.num_experts)
            ]
        )

        self.shared_expert = TimeMoeTemporalBlock(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """
        Forward pass through the sparse experts layer.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Transformed hidden states and router logits
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits -> (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)  # noqa: E501

        # Loop over all available experts in the model and perform the computation on
        # each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )  # noqa: E501

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )  # noqa: E501

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )  # noqa: E501

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )  # noqa: E501
        return final_hidden_states, router_logits


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2Attention with
# Qwen2->TimeMoe
class TimeMoeAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: TimeMoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx`"
                "is not recommended and will to errors during the forward call,"
                "if caching is used. Please make sure to provide a `layer_idx`, when"
                "creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`:"
                f" {self.hidden_size} and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )  # noqa: E501
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )  # noqa: E501
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )  # noqa: E501
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )  # noqa: E501

        self.rotary_emb = TimeMoeRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """
        Forward pass through the attention layer.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor
        attention_mask : Optional[torch.Tensor]
            Mask to avoid attending to padding tokens
        position_ids : Optional[torch.LongTensor]
            Position indices for positional embeddings
        past_key_value : Optional[Cache]
            Cached key/value states for faster decoding
        output_attentions : bool
            Whether to return attention weights
        kwargs : dict
            Additional arguments

        Returns
        -------
        tuple
            Output states, attention weights (optional), and cached states (optional)
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37."
                "Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # noqa: E501
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # noqa: E501
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # noqa: E501

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "  # noqa: E501
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "  # noqa: E501
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"  # noqa: E501
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"  # noqa: E501
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"  # noqa: E501
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


TIME_MOE_ATTENTION_CLASSES = {
    "eager": TimeMoeAttention,
}


class TimeMoeDecoderLayer(nn.Module):
    """
    TimeMoeDecoderLayer is a custom decoder layer for the TimeMOE model.

    Parameters
    ----------
    config : TimeMoeConfig
        Configuration object containing model hyperparameters.
    layer_idx : int
        Index of the current layer.
    """

    def __init__(self, config: TimeMoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = TIME_MOE_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx
        )

        if self.config.use_dense:
            self.ffn_layer = TimeMoeMLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
            )
        else:
            self.ffn_layer = TimeMoeSparseExpertsLayer(config)
        self.input_layernorm = TimeMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = TimeMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        Optional[torch.FloatTensor],
        Optional[torch.FloatTensor],
    ]:
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input to the layer of shape (batch, seq_len, embed_dim).
        attention_mask : Optional[torch.Tensor], optional
            Attention mask of size (batch, sequence_length) where padding elements are indicated by 0, by default None.
        position_ids : Optional[torch.LongTensor], optional
            Positional encoding indices, by default None.
        past_key_value : Optional[tuple[torch.Tensor]], optional
            Cached past key and value projection states, by default None.
        output_attentions : Optional[bool], optional
            Whether or not to return the attention tensors of all attention layers, by default False.
        use_cache : Optional[bool], optional
            If set to True, past_key_values key value states are returned and can be used to speed up decoding, by default False.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]
            A tuple containing:

            - hidden_states (torch.FloatTensor): The output hidden states.
            - self_attn_weights (torch.FloatTensor): The attention weights.
            - present_key_value (Optional[torch.FloatTensor]): The present key value states.
            - router_logits (Optional[torch.FloatTensor]): The router logits.
        """  # noqa: E501
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.ffn_layer(hidden_states)
        hidden_states = residual + hidden_states

        if not output_attentions:
            self_attn_weights = None

        if not use_cache:
            present_key_value = None
        return hidden_states, self_attn_weights, present_key_value, router_logits


class TimeMoePreTrainedModel(PreTrainedModel):
    """
    TimeMoePreTrainedModel is a pre-trained model class for TimeMoe.

    Inherits from PreTrainedModel.

    Parameters
    ----------
    config_class : class
        The configuration class to use for this model.
    base_model_prefix : str
        The prefix for the base model.
    supports_gradient_checkpointing : bool
        Indicates if the model supports gradient checkpointing.
    _no_split_modules : list
        List of module names that should not be split.
    _skip_keys_device_placement : str
        Keys to skip during device placement.
    _supports_flash_attn_2 : bool
        Indicates if the model supports flash attention version 2.
    _supports_sdpa : bool
        Indicates if the model supports scaled dot-product attention.
    _supports_cache_class : bool
        Indicates if the model supports cache class.
    """

    config_class = TimeMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TimeMoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class TimeMoeModel(TimeMoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.

    Each layer is a [`TimeMoeDecoderLayer`].

    Parameters
    ----------
        config: TimeMoeConfig
    """

    def __init__(self, config: TimeMoeConfig):
        super().__init__(config)
        self.embed_layer = TimeMoeInputEmbedding(config)
        self.layers = nn.ModuleList(
            [
                TimeMoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = TimeMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MoeModelOutputWithPast]:
        """
        Perform a forward pass of the model.

        Parameters
        ----------
        input_ids : torch.FloatTensor, optional
            Input tensor of shape [batch_size, seq_len, input_size] representing the
            input time series.
        attention_mask : torch.Tensor, optional
            Attention mask tensor.
        position_ids : torch.LongTensor, optional
            Tensor of position IDs.
        past_key_values : list of torch.FloatTensor, optional
            List of past key values for caching.
        inputs_embeds : torch.FloatTensor, optional
            Input embeddings tensor.
        use_cache : bool, optional
            Whether to use caching.
        output_attentions : bool, optional
            Whether to output attention weights.
        output_hidden_states : bool, optional
            Whether to output hidden states.
        return_dict : bool, optional
            Whether to return a dictionary instead of a tuple.

        Returns
        -------
        Union[tuple, MoeModelOutputWithPast]
            If `return_dict` is True, returns a `MoeModelOutputWithPast` object
            containing:

                - last_hidden_state: torch.FloatTensor
                    The last hidden state of the model.
                - past_key_values: list of torch.FloatTensor
                    The past key values for caching.
                - hidden_states: tuple of torch.FloatTensor, optional
                    The hidden states of the model.
                - attentions: tuple of torch.FloatTensor, optional
                    The attention weights.
                - router_logits: tuple of torch.FloatTensor
                    The router logits.
            If `return_dict` is False, returns a tuple containing:

                - last_hidden_state: torch.FloatTensor
                    The last hidden state of the model.
                - past_key_values: list of torch.FloatTensor
                    The past key values for caching.
                - hidden_states: tuple of torch.FloatTensor, optional
                    The hidden states of the model.
                - attentions: tuple of torch.FloatTensor, optional
                    The attention weights.
                - router_logits: tuple of torch.FloatTensor
                    The router logits.
        """
        # input_ids is the input of time series, its shape is [batch_size, seq_len,
        # input_size]
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

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"  # noqa: E501
            )
        elif input_ids is not None:
            if len(input_ids.shape) == 2:
                input_ids.unsqueeze_(dim=-1)
            batch_size, seq_length, _ = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                warnings.warn(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."  # noqa: E501
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            position_ids = position_ids.view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_layer(input_ids)

        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=None,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = ()
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            all_router_logits += (layer_outputs[-1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache = layer_outputs[2]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
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
                    all_router_logits,
                ]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class TimeMoeOutputLayer(nn.Module):
    """
    TimeMoeOutputLayer applies a linear transformation to the input data.

    Parameters
    ----------
    hidden_size : int
        The size of the hidden layer.
    horizon_length : int
        The length of the prediction horizon.
    input_size : int, optional
        The size of the input data, by default 1.
    """

    def __init__(self, hidden_size: int, horizon_length: int, input_size: int = 1):
        super().__init__()

        self.out_layer = nn.Linear(
            hidden_size,
            input_size * horizon_length,
            bias=False,
        )

    def forward(self, x):
        """
        Forward pass through the output layer.

        Parameters
        ----------
        x: torch.FloatTensor
            With shape [B, seq_len, hidden_size]

        Returns
        -------
        torch.FloatTensor:
            Final prediction with shape [B, seq_len, input_size]
        """
        return self.out_layer(x)


class TimeMoeForPrediction(TimeMoePreTrainedModel, TSGenerationMixin):
    """
    TimeMoeForPrediction is a model class designed for time series prediction tasks.

    It extends the TimeMoePreTrainedModel and TSGenerationMixin classes,
    providing functionalities for training and inference with multiple experts and
    auxiliary loss.

    Parameters
    ----------
    config : TimeMoeConfig
        Configuration object containing model hyperparameters and settings.
    """

    def __init__(self, config: TimeMoeConfig):
        super().__init__(config)
        self.config = config
        self.apply_aux_loss = config.apply_aux_loss
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_factor = config.router_aux_loss_factor

        self.model = TimeMoeModel(config)
        # output layer
        lm_head_list = []
        self.horizon_length_map = {}
        for i, horizon_length in enumerate(config.horizon_lengths):
            lm_head_list.append(
                TimeMoeOutputLayer(
                    hidden_size=self.config.hidden_size,
                    input_size=self.config.input_size,
                    horizon_length=horizon_length,
                )
            )
            self.horizon_length_map[horizon_length] = i
        self.lm_heads = nn.ModuleList(lm_head_list)

        self.loss_function = torch.nn.HuberLoss(reduction="none", delta=2.0)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        """
        Set the decoder model for the instance.

        Parameters
        ----------
        decoder : object
            The decoder model to be set.
        """
        self.model = decoder

    def get_decoder(self):
        """
        Retrieve the decoder model.

        Returns
        -------
            object: The decoder model instance.
        """
        return self.model

    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        loss_masks: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        max_horizon_length: Optional[int] = None,
    ) -> Union[tuple, MoeCausalLMOutputWithPast]:
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        input_ids : torch.FloatTensor, optional
            Input tensor containing token IDs.
        attention_mask : Optional[torch.Tensor], optional
            Mask to avoid performing attention on padding token indices.
        position_ids : Optional[torch.LongTensor], optional
            Indices of positions of each input sequence tokens in the batch.
        past_key_values : Optional[list[torch.FloatTensor]], optional
            Contains precomputed key and value hidden states of the attention blocks.
        inputs_embeds : Optional[torch.FloatTensor], optional
            Optionally, instead of passing input_ids you can choose to directly pass an
            embedded representation.
        labels : Optional[torch.FloatTensor], optional
            Labels for computing the language modeling loss.
        loss_masks : Optional[torch.FloatTensor], optional
            Masks to apply on the loss computation.
        use_cache : Optional[bool], optional
            If set to `True`, `past_key_values` key value states are returned and can be
            used to speed up decoding.
        output_attentions : Optional[bool], optional
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states : Optional[bool], optional
            Whether or not to return the hidden states of all layers.
        return_dict : Optional[bool], optional
            Whether or not to return a `MoeCausalLMOutputWithPast` instead of a plain
            tuple.
        max_horizon_length : Optional[int], optional
            Maximum horizon length for predictions.

        Returns
        -------
        Union[tuple, MoeCausalLMOutputWithPast]
            A tuple or `MoeCausalLMOutputWithPast` containing the model outputs.
        """
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        predictions = None

        loss = None
        aux_loss = None
        if labels is not None:
            # AutoRegressive loss
            ar_loss = 0.0
            for lm_head, horizon_length in zip(
                self.lm_heads, self.config.horizon_lengths
            ):
                one_predictions = lm_head(hidden_states)
                one_loss = self.calc_ar_loss(
                    one_predictions, labels, loss_masks, horizon_length
                )
                ar_loss += one_loss
                if predictions is None:
                    predictions = one_predictions
            loss = ar_loss / len(self.config.horizon_lengths)

            if self.apply_aux_loss:
                router_logits = outputs.router_logits if return_dict else outputs[-1]

                temporal_aux_loss = load_balancing_loss_func(
                    router_logits,
                    top_k=self.num_experts_per_tok,
                    num_experts=self.config.num_experts,
                    attention_mask=attention_mask,
                )
                loss += self.router_aux_loss_factor * temporal_aux_loss.to(loss.device)
        else:
            if max_horizon_length is None:
                horizon_length = self.config.horizon_lengths[0]
                max_horizon_length = horizon_length
            else:
                horizon_length = self.config.horizon_lengths[0]
                for h in self.config.horizon_lengths[1:]:
                    if h > max_horizon_length:
                        break
                    else:
                        horizon_length = h
            lm_head = self.lm_heads[self.horizon_length_map[horizon_length]]
            predictions = lm_head(hidden_states)
            if horizon_length > max_horizon_length:
                predictions = predictions[
                    :, :, : self.config.input_size * max_horizon_length
                ]

        if not return_dict:
            output = (predictions,) + outputs[1:]
            return (loss, aux_loss) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=predictions,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def calc_ar_loss(self, predictions, labels, loss_masks, horizon_length):
        """Calculate the autoregressive loss for the given predictions and labels."""
        if len(labels.shape) == 2:
            labels.unsqueeze_(dim=-1)
            # enable model parallelism
            labels = labels.to(predictions.device)
        if loss_masks is not None and len(loss_masks.shape) == 2:
            loss_masks.unsqueeze_(dim=-1)
            # enable model parallelism
            loss_masks = loss_masks.to(predictions.device)

        if horizon_length > 1:
            batch_size, seq_len, output_size = predictions.shape
            shift_predictions = predictions.view(
                batch_size, seq_len, horizon_length, -1
            )

            # pad to the same length with predictions
            # shape -> [B, input_size, seq_len + horizon_length -1]
            labels = F.pad(
                labels.transpose(-1, -2),
                (0, horizon_length - 1),
                mode="constant",
                value=0,
            )

            # shape -> [B, input_size, seq_len, horizon_length]
            shift_labels = labels.unfold(dimension=-1, size=horizon_length, step=1)
            shift_labels = shift_labels.permute(0, 2, 3, 1)

            if loss_masks is not None:
                # pad to the same length with predictions
                loss_masks = F.pad(
                    loss_masks.transpose(-1, -2),
                    (0, horizon_length - 1),
                    mode="constant",
                    value=0,
                )

                loss_masks = loss_masks.unfold(
                    dimension=-1, size=horizon_length, step=1
                )
                loss_masks = loss_masks.permute(0, 2, 3, 1)

        else:
            shift_predictions = predictions
            shift_labels = labels

        # Calculate loss with mask
        losses = self.loss_function(shift_predictions, shift_labels)

        if loss_masks is not None:
            losses = losses * loss_masks
            loss = losses.sum() / loss_masks.sum()
        else:
            loss = torch.mean(losses)

        return loss

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation by handling past key values, attention masks, and input embeddings.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input token IDs.
        past_key_values : Optional[Union[Cache, List[torch.Tensor]]], default=None
            The past key values used for caching in generation. Can be an instance of
            `Cache` or a list of tensors.
        attention_mask : Optional[torch.Tensor], default=None
            The attention mask to avoid performing attention on padding token indices.
        inputs_embeds : Optional[torch.Tensor], default=None
            The input embeddings. If provided, they are used in the first generation
            step.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing the prepared inputs for generation, including `input_ids`, `position_ids`,
            `past_key_values`, `use_cache`, and `attention_mask`.
        """  # noqa: E501
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                if isinstance(past_key_values, DynamicCache):
                    past_length = past_key_values.seen_tokens
                else:
                    past_length = cache_length

                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids,
            # then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache
            # (e.g. when passing input_embeds as input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard  # noqa: E501
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids
            # only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the
            # input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation
        # step.
        if inputs_embeds is not None and past_key_values is None:
            warnings.warn("Use input_embedding")
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
