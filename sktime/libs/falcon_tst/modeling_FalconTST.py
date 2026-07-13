"""PyTorch model definitions for Falcon-TST forecasting models."""

import math
from functools import reduce

from skbase.utils.dependencies import _check_soft_dependencies

# import transformer_engine as te
from .configuration_FalconTST import FalconTSTConfig

if _check_soft_dependencies("torch", "transformers", severity="none"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from transformers import PreTrainedModel
else:

    class _FalconTSTDependencyStub:
        """Placeholder base used when Falcon-TST dependencies are unavailable."""

    class _FalconTSTNNStub:
        """Minimal ``torch.nn`` placeholder for import-time class definitions."""

        Module = object

    class _FalconTSTTorchStub:
        """Minimal ``torch`` placeholder for import-time annotations."""

        Tensor = object
        nn = _FalconTSTNNStub

        @staticmethod
        def no_grad():
            """Return a no-op decorator for import-time method definitions."""

            def decorator(func):
                return func

            return decorator

    torch = _FalconTSTTorchStub()
    F = None
    nn = _FalconTSTNNStub
    Tensor = object
    PreTrainedModel = _FalconTSTDependencyStub


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even].

    Args:
        x (Tensor): Input tensor

    Returns
    -------
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape
            [seq_length, ..., dim]

    Returns
    -------
        Tensor: The input tensor after applying RoPE
    """
    freqs = freqs.to(t.device)
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Embedding.

    Args:
        kv_channels (int): Projection weights dimension in multi-head
            attention, obtained from transformer config.
        rotary_interleaved (bool, optional): If True, interleaved rotary
            position embeddings. Defaults to False.
        rotary_base (int, optional): Base period for rotary position embeddings.
            Defaults to 10000.
        use_cpu_initialization (bool, optional): If False, initialize the
            inv_freq directly on the GPU. Defaults to False.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_interleaved: bool = False,
        rotary_base: int = 10000,
        use_cpu_initialization: bool = False,
    ) -> None:
        super().__init__()

        dim = kv_channels
        self.rotary_interleaved = rotary_interleaved
        if use_cpu_initialization or not torch.cuda.is_available():
            device = "cpu"
        else:
            device = torch.cuda.current_device()
        self.inv_freq = 1.0 / (
            rotary_base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

    def get_freqs_non_repeated(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Generate matrix of frequencies based on positions in the sequence."""
        seq = (
            torch.arange(
                max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            + offset
        )
        freqs = torch.outer(seq, self.inv_freq)  # [seq len, dim]
        return freqs

    def forward(
        self, max_seq_len: int, offset: int = 0, packed_seq: bool = False, device=None
    ) -> Tensor:
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.
            packed_seq (bool, optional): Whether to use packed sequence.
                Defaults to False.

        Returns
        -------
            Tensor: Embeddings after applying RoPE.
        """
        if device is None:
            device = self.inv_freq.device
        if self.inv_freq.device.type == "cpu":
            # move `inv_freq` to GPU once at the first micro-batch forward pass
            self.inv_freq = self.inv_freq.to(device=device)

        freqs = self.get_freqs_non_repeated(max_seq_len, offset).to(device)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        return emb.to(device)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f"{prefix}inv_freq", None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_rotary_seq_len(
        self,
        transformer_input: Tensor,
    ) -> float:
        """Return the rotary sequence length.

        Args:
            transformer_input (Tensor): Input tensor to the transformer

        Returns
        -------
            float: The rotary sequence length
        """
        rotary_seq_len = transformer_input.size(0)
        return rotary_seq_len


class IdentityOp(nn.Module):
    """Identity operation."""

    def forward(self, x):
        """Return input unchanged."""
        return x


class RMSNorm(nn.Module):
    """Root mean square normalization layer."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """Normalize hidden states of shape [bs, patch_num, d_model]."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TEDotProductAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, q, k, v, attention_mask):
        """Implement the multihead softmax attention.

        Arguments
        ---------
            q,k,v: The tensor containing the query, key, and value.
                Shape: [seq_len, batch_size, hidden_size]
            attention_mask: boolean mask to apply to the attention weights.
                True means to keep, False means to mask out.
                Shape: [batch_size, 1, seq_len, seq_len]
        """
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()

        batch_size, seq_len = q.shape[0], q.shape[1]
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        # scores
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        scores = scores.masked_fill(attention_mask == 0, float("-1e9"))
        # Softmax
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        # Dropout
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        output = output.reshape(batch_size, seq_len, -1)

        output = output.transpose(0, 1).contiguous()
        return output


class SelfAttention(nn.Module):
    """Self-attention block."""

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.core_attention = TEDotProductAttention()
        self.linear_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=config.add_bias_linear,
        )
        self.linear_qkv = nn.Linear(
            self.hidden_size,
            3 * self.hidden_size,
            bias=config.add_bias_linear,
        )

    def forward(self, x, attention_mask, rotary_pos_emb):
        """Apply self-attention to hidden states."""
        qkv = self.linear_qkv(x)
        qkv = qkv.view(qkv.size(0), qkv.size(1), self.config.num_attention_heads, -1)
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply rotary encoding to q and k
        rotary_pos_emb = (rotary_pos_emb,) * 2
        q_pos_emb, k_pos_emb = rotary_pos_emb
        q = _apply_rotary_pos_emb_bshd(q, q_pos_emb)
        k = _apply_rotary_pos_emb_bshd(k, k_pos_emb)

        # attention
        attn_output = self.core_attention(q, k, v, attention_mask)
        output = self.linear_proj(attn_output)
        return output


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config, in_features):
        super().__init__()
        self.config = config
        self.linear_fc1 = nn.Linear(
            in_features,
            self.config.moe_ffn_hidden_size * 2,
            bias=self.config.add_bias_linear,
        )
        self.linear_fc2 = nn.Linear(
            self.config.moe_ffn_hidden_size,
            self.config.hidden_size,
            bias=self.config.add_bias_linear,
        )

    def forward(self, x):
        """Apply feed-forward transformation."""
        x = self.swiglu(self.linear_fc1(x))
        x = self.linear_fc2(x)
        return x

    def swiglu(self, y):
        """Perform SwiGLU activation.

        Args:
            y (torch.Tensor): Input tensor to split into two halves along the
                last dimension.

        Returns
        -------
            torch.Tensor: Result of SwiGLU activation.
        """
        y_1, y_2 = torch.chunk(y, 2, -1)
        return F.silu(y_1) * y_2


class TransformerLayer(nn.Module):
    """Transformer layer."""

    def __init__(self, config, input_layernorm):
        super().__init__()
        self.config = config
        if input_layernorm:
            self.input_layernorm = RMSNorm(self.config.hidden_size)
        else:
            self.input_layernorm = IdentityOp()
        self.self_attention = SelfAttention(config)
        self.pre_mlp_layernorm = RMSNorm(self.config.hidden_size)
        self.mlp = MLP(config, self.config.hidden_size)

    def forward(self, x, attention_mask, rotary_pos_emb):
        """Apply one transformer layer."""
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attention(x, attention_mask, rotary_pos_emb)
        x = x + residual
        residual = x
        x = self.pre_mlp_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x


class FalconTSTExpert(nn.Module):
    """Falcon-TST expert module."""

    def __init__(
        self, config, patch_input_size=32, expert_output_size=336, final_layernorm=True
    ):
        super().__init__()
        self.config = config
        self.patch_size = patch_input_size
        self.seq_length = config.seq_length
        assert self.seq_length % self.patch_size == 0, (
            f"invalid patch_size: {self.patch_size} when seq_length={self.seq_length}"
        )
        self.patch_num = self.seq_length // self.patch_size
        self.flatten_size = self.patch_num * self.config.hidden_size

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    config, input_layernorm=config.transformer_input_layernorm
                )
                for _ in range(self.config.expert_num_layers)
            ]
        )
        if final_layernorm:
            self.final_layernorm = RMSNorm(self.config.hidden_size)
        else:
            self.final_layernorm = IdentityOp()
        self.patch_embedding = MLP(config, in_features=patch_input_size)
        self.output_layer = nn.Linear(
            in_features=self.flatten_size,
            out_features=expert_output_size,
            bias=False,
        )

    def _forward_patch_embedding(
        self,
        input: Tensor,  # [batch_size, seq_len]
    ):
        """
        Perform patch embedding on the input time series.

        This method applies a linear transformation to the input tensor to
        convert it into patches and then embeds these patches using a linear layer.
        """
        batch_size, seq_len = input.shape
        assert seq_len == self.seq_length, (
            f"Expected sequence length {self.seq_length}, but got {seq_len}"
        )

        # Create input_mask based on pad_length
        # When a time point is masked, its value is mask_pad_value(default:255.)
        input_mask = (
            input != self.config.mask_pad_value
        )  # 0: mask, 1: unmask   [batch_size, seq_len]

        # so whether the masked value 0 has the same effective of attention_mask
        input_data = input * input_mask  # [batch_size, seq_len]

        # Patchify the input
        input_data = input_data.unfold(
            dimension=-1, size=self.patch_size, step=self.patch_size
        ).contiguous()  # input [batch_size, patch_num, patch_size]
        hidden_states = self.patch_embedding(
            input_data
        )  # hidden_states [batch_size, patch_num, hidden_size]
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        # hidden_states: [patch_num, batch_size, hidden_size]

        # Patchify mask: only fully masked patches are masked.
        attention_mask = input_mask.unfold(
            dimension=-1, size=self.patch_size, step=self.patch_size
        ).contiguous()  # [batch_size, patch_num, patch_size]
        attention_mask = (
            attention_mask.sum(-1) == self.patch_size
        )  # [batch_size, patch_num]   # 0: mask, 1: unmask
        attention_mask[:, -1] = True  # The last patch is not masked
        _, patch_num = attention_mask.shape
        attention_mask = attention_mask.unsqueeze(2).repeat(
            1, 1, patch_num
        ) * attention_mask.unsqueeze(1).repeat(
            1, patch_num, 1
        )  # [batch_size, patch_num, patch_num]
        attention_mask = attention_mask.unsqueeze(
            1
        ).contiguous()  # [batch_size, 1, patch_num, patch_num]

        return hidden_states, attention_mask, input_mask

    def _forward_output(self, hidden_states, output_scale=None, input_mask=None):
        """Perform a forward pass through the output layer.

        Args:
            hidden_states (Tensor): Transformed hidden states of shape
                [patch_num, batch_size, hidden_size].
            output_scale (Tensor, optional): Expert probabilities for the
                output layer [batch_size].
            input_mask (Tensor, optional): Expert input mask of shape
                [batch_size, seq_len], where 0 means mask and 1 means unmask.

        Returns
        -------
            expert_output (Tensor): Expert output of shape
                [batch_size, expert_output_size].
        """
        # [patch_num, batch_size, hidden_size] ->
        # [batch_size, flatten_size (patch_num * hidden_size)]
        patch_num, batch_size, hidden_size = hidden_states.shape
        assert (patch_num * hidden_size) == self.flatten_size, (
            f"patch_num ({patch_num}) * hidden_size ({hidden_size}) != "
            f"flatten_size ({self.flatten_size})"
        )
        hidden_states = (
            hidden_states.transpose(0, 1).reshape(-1, self.flatten_size).contiguous()
        )
        expert_output = self.output_layer(
            hidden_states
        )  # [batch_size, expert_output_size]
        if output_scale is not None:
            original_dtype = expert_output.dtype
            expert_output = expert_output * output_scale.unsqueeze(-1)
            expert_output = expert_output.to(original_dtype)

        return expert_output

    def forward(self, expert_input, rotary_pos_emb, expert_probs=None):
        """Run expert forward pass."""
        hidden_states, attention_mask, input_mask = self._forward_patch_embedding(
            expert_input
        )
        # hidden_states:  [patch_num, batch_size, hidden_size]
        # attention_mask: [batch_size, 1, patch_num, patch_num]
        # input_mask:     [batch_size, seq_len]

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, rotary_pos_emb[: hidden_states.shape[0]]
            )

        hidden_states = self.final_layernorm(hidden_states)

        expert_output = self._forward_output(hidden_states, expert_probs, input_mask)
        return expert_output


class SequentialFalconTST(nn.Module):
    """Sequential container for Falcon-TST experts."""

    def __init__(self, config, expert_output_size=336):
        super().__init__()
        self.config = config
        self.expert_output_size = expert_output_size
        self.local_experts = nn.ModuleList(
            [
                FalconTSTExpert(
                    config,
                    expert_output_size=expert_output_size,
                    patch_input_size=config.patch_size_list[expert_id],
                    final_layernorm=config.moe_expert_final_layernorm,
                )
                for expert_id in range(config.num_moe_experts)
            ]
        )

    def forward(self, input, routing_map, rotary_pos_emb, expert_probs):
        """Dispatch inputs to local experts and combine their outputs."""
        expert_output_list = []
        batch_size, seq_len = input.size()

        for i, expert in enumerate(self.local_experts):
            token_mask = routing_map[:, i].bool()  # shape (batch,)
            current_inputs = input[token_mask]  # (num_tokens_for_expert, seq_len)
            current_probs = expert_probs[token_mask, i]

            if current_inputs.numel() == 0:
                expert_output = torch.zeros(
                    0, self.expert_output_size, device=input.device, dtype=input.dtype
                )
            else:
                expert_output = expert(current_inputs, rotary_pos_emb, current_probs)

            full_output = torch.zeros(
                batch_size,
                self.expert_output_size,
                device=input.device,
                dtype=input.dtype,
            )
            full_output[token_mask] = expert_output
            expert_output_list.append(full_output)

        expert_output = reduce(torch.add, expert_output_list)
        return expert_output


class TopKRouter(nn.Module):
    """Top-k router for Falcon-TST experts."""

    def __init__(self, config: FalconTSTConfig):
        super().__init__()
        self.config = config
        self.topk = config.moe_router_topk

        self.weight = nn.Parameter(
            torch.empty(
                (config.num_moe_experts, config.moe_router_input_size),
                dtype=torch.float32,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reset router parameters."""
        nn.init.normal_(self.weight, mean=0, std=self.config.init_method_std)

    def routing(self, logits: torch.Tensor):
        """Route logits to top-k experts."""
        score_function = self.config.moe_router_score_function

        if score_function == "softmax":
            if self.config.moe_router_pre_softmax:
                scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(
                    logits
                )
                probs, top_indices = torch.topk(scores, self.topk, dim=1)
            else:
                scores, top_indices = torch.topk(logits, self.topk, dim=1)
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(
                    logits
                )
        else:
            raise NotImplementedError

        routing_probs = torch.zeros_like(logits).scatter_(1, top_indices, probs)
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            1, top_indices, True
        )

        return routing_probs, routing_map

    def forward(self, input: torch.Tensor):
        """Compute routing probabilities and map."""
        if self.weight.device != input.device:
            self.weight.data = self.weight.data.to(input.device)

        gating_logits = F.linear(input, self.weight)
        num_tokens = gating_logits.shape[:-1].numel()
        gating_logits = gating_logits.view(num_tokens, self.config.num_moe_experts)

        scores, routing_map = self.routing(gating_logits)

        return scores, routing_map


class FalconTSTMoELayer(nn.Module):
    """Mixture-of-experts layer for Falcon-TST."""

    def __init__(self, config, layer_number):
        super().__init__()
        self.config = config
        self.seq_length = config.seq_length
        self.router = TopKRouter(config)
        self.layer_number = layer_number
        self.pred_length = config.pred_length
        self.is_last_layer = self.layer_number == config.num_hidden_layers
        if self.is_last_layer and self.config.heterogeneous_moe_layer:
            self.expert_output_size = config.pred_length
        else:
            if self.config.do_expert_forecast:
                self.expert_output_size = config.seq_length + config.pred_length
            else:
                self.expert_output_size = config.seq_length

        if self.is_last_layer and self.config.heterogeneous_moe_layer:
            # If heterogeneous_moe_layer is True, the backcast will be None
            self.backcast_layernorm = None
        else:
            self.backcast_layernorm = RMSNorm(self.seq_length)

        self.experts = SequentialFalconTST(
            config,
            expert_output_size=self.expert_output_size,
        )
        self.shared_experts = FalconTSTExpert(
            config,
            expert_output_size=self.expert_output_size,
            patch_input_size=config.shared_patch_size,
            final_layernorm=config.moe_expert_final_layernorm,
        )

    def time_series_preprocess(self, input: torch.Tensor):
        """Preprocess time series sample for dispatch.

        Applies RevIN to input time series samples, and processes the input
        mask, where 0 means mask and 1 means unmask.

        Args:
            input (torch.Tensor): Input time series samples to the MoE layer.
                Shape: [batch_size, seq_len].

        Returns
        -------
            input (torch.Tensor): Backcast time series samples.
        """
        batch_size, seq_len = input.shape
        assert seq_len == self.seq_length, (
            f"seq_len {seq_len} != self.seq_length {self.seq_length}"
        )

        # Create input_mask based on pad_length
        # When a time point is masked, its value is mask_pad_value(default:255.)
        input_mask = (
            input != self.config.mask_pad_value
        )  # 0: mask, 1: unmask   [batch_size, seq_len]

        self.input_mask = input_mask

        return input

    def router_and_preprocess(self, backcast: torch.Tensor):
        """Compute and preprocess time series sample routing for dispatch.

        This method uses the router to determine which experts to send each
        time series sample to, producing routing probabilities and a mapping.
        The original input time series samples are returned as a residual
        connection.
        """
        # backcast [batch_size, seq_len]    means/stdev [batch_size, 1]
        backcast = self.time_series_preprocess(backcast)

        # residual: [batch_size, seq_len], the input to the shared experts
        residual = backcast

        # TODO: Check the effective of the masked value to the router
        probs, routing_map = self.router(
            backcast * self.input_mask
        )  # probs/routing_map: [batch_size, num_experts]

        return backcast, probs, residual, routing_map

    def experts_compute(
        self,
        input: torch.Tensor,  # [num_permuted_samples_after_dispatch, seq_len]
        probs: torch.Tensor,  # [num_permuted_samples_after_dispatch]
        residual: torch.Tensor,  # [batch_size, seq_len]
        rotary_pos_emb: torch.Tensor,
        routing_map: torch.Tensor,
    ):
        """Compute expert outputs on dispatched time series samples.

        This method first post-processes the dispatched input to get permuted
        time series samples for each expert. It then passes the samples through
        the local experts. If a shared expert is configured and not overlapped
        with communication, it is also applied.
        """
        # shared_expert_output: [batch_size, seq_len (+ pred_len)]
        shared_experts_output = self.shared_experts(residual, rotary_pos_emb)

        # dispatched_input (global_input_tokens):
        # [num_permuted_samples_after_dispatch_postprocess(sorted), seq_len]
        # tokens_per_expert (global_probs):         [num_experts]
        # permuted_probs (global_probs):
        # [num_permuted_samples_after_dispatch_postprocess(sorted)]

        experts_output = self.experts(input, routing_map, rotary_pos_emb, probs)

        return experts_output, shared_experts_output

    def combine(
        self,
        experts_output: torch.Tensor,
        shared_experts_output: torch.Tensor,
    ):
        """Combine expert outputs and add shared expert output.

        This method uses the time series sample dispatcher to combine the
        outputs from different experts. It then adds the output from the shared
        expert if it exists.
        """
        assert experts_output.shape == shared_experts_output.shape, (
            f"experts_output shape {experts_output.shape} doesn't equal "
            f"shared_experts_output shape:{shared_experts_output.shape}"
        )
        output = experts_output + shared_experts_output

        if self.is_last_layer and self.config.heterogeneous_moe_layer:
            output_backcast = None
            output_forecast = output
            assert output_forecast.shape[1] == self.pred_length, (
                "heterogeneous_moe_layer=True, expected the last moe layer's "
                f"output pred len: {self.pred_length}, but got "
                f"{output_forecast.shape[1]}"
            )
        else:
            # The masked time point may not be mask_pad_value(default:255.);
            # it will be postprocessed.
            output_backcast = output[:, : self.seq_length]  # [batch_size, seq_len]

            if self.config.do_expert_forecast:
                output_forecast = output[:, self.seq_length :]  # [batch_size, pred_len]
                assert output_forecast.shape[1] == self.pred_length, (
                    "do_expert_forecast=True, expected the last moe layer's "
                    f"output pred len: {self.pred_length}, but got "
                    f"{output_forecast.shape[1]}"
                )
            else:
                output_forecast = None

        return output_backcast, output_forecast

    def postprocess(
        self,
        backcast: torch.Tensor,  # [batch_size, seq_len]
        forecast: torch.Tensor,  # [batch_size, pred_len]
        output_backcast: torch.Tensor,  # [batch_size, seq_len]
        output_forecast: torch.Tensor,  # [batch_size, pred_len]
    ):
        """Postprocess MoE layer outputs.

        Args:
            backcast (torch.Tensor): Previous layer backcast samples.
            forecast (torch.Tensor): Previous layer forecast samples.
            output_backcast (torch.Tensor): Current layer backcast samples.
            output_forecast (torch.Tensor): Current layer forecast samples.
        """
        if output_backcast is not None:
            # 25/8/14 @modified by xiaming replace the revin with layernorm
            # after the moe layer. Multiplying output_backcast with the input
            # mask hurts performance.
            output_backcast = self.backcast_layernorm(output_backcast)  # LayerNorm
            if self.config.residual_backcast:
                output_backcast = backcast - output_backcast

            output_backcast[~self.input_mask] = self.config.mask_pad_value
            # Recover masked time points back to mask_pad_value(default:255.).

        if (
            self.config.do_expert_forecast and forecast is not None
        ):  # The first layer's forecast is None
            output_forecast = forecast + output_forecast

        return output_backcast, output_forecast

    def forward(self, backcast, forecast, rotary_pos_emb):
        """Run MoE layer forward pass."""
        inputs, probs, residual, routing_map = self.router_and_preprocess(backcast)
        experts_output, shared_experts_output = self.experts_compute(
            inputs, probs, residual, rotary_pos_emb, routing_map
        )
        output_backcast, output_forecast = self.combine(
            experts_output, shared_experts_output
        )
        output_backcast, output_forecast = self.postprocess(
            backcast, forecast, output_backcast, output_forecast
        )
        return output_backcast, output_forecast


class FalconTSTBlock(nn.Module):
    """Falcon-TST block."""

    def __init__(self, config, input_layernorm=True):
        super().__init__()
        self.config = config

        if input_layernorm:
            self.input_layernorm = RMSNorm(self.config.seq_length)
        else:
            self.input_layernorm = IdentityOp()

        self.layers = nn.ModuleList(
            [
                FalconTSTMoELayer(config, layer_num + 1)
                for layer_num in range(self.config.num_hidden_layers)
            ]
        )

    def forward(self, x, rotary_pos_emb):
        """Run block forward pass."""
        backcast = x
        forecast = None

        input_mask = backcast != self.config.mask_pad_value
        backcast = self.input_layernorm(backcast * input_mask)
        backcast[~input_mask] = self.config.mask_pad_value

        for layer in self.layers:
            backcast, forecast = layer(backcast, forecast, rotary_pos_emb)
        return backcast, forecast


class FalconTSTPreTrainedModel(PreTrainedModel):
    """Falcon-TST pretrained model base class."""

    config_class = FalconTSTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FalconTSTMoELayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class FalconTSTModel(FalconTSTPreTrainedModel):
    """Falcon-TST backbone model."""

    def __init__(self, config: FalconTSTConfig):
        super().__init__(config)
        self.config = config
        self.seq_length = self.config.seq_length
        self.rotary_pos_emb = RotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_base=self.config.rotary_base,
            use_cpu_initialization=self.config.use_cpu_initialization,
            rotary_interleaved=self.config.rotary_interleaved,
        )
        self.decoder = FalconTSTBlock(
            config=config, input_layernorm=self.config.block_input_layernorm
        )
        if self.config.do_expert_forecast and self.config.heterogeneous_moe_layer:
            self.output_layer = IdentityOp()
        else:
            self.output_layer = nn.Linear(
                in_features=self.seq_length,
                out_features=self.config.pred_length,
                bias=self.config.add_bias_linear,
            )

    def revin(
        self,
        input: Tensor,  # [batch_size, seq_len]
        input_mask: Tensor,  # [batch_size, seq_len] 0:mask, 1:unmask
    ):
        """Normalize input using Non-stationary Transformer normalization."""
        input_data = input * input_mask
        sum_per_sample = torch.sum(
            input_data, dim=1, keepdim=True
        ).detach()  # [batch_size, 1], torch.bfloat16
        count_per_sample = torch.sum(
            input_mask, dim=1, keepdim=True
        ).detach()  # [batch_size, 1], torch.int64
        assert not torch.any(count_per_sample == 0), (
            "There is zero in count_per_sample, shape: "
            f"{input[torch.where(count_per_sample.squeeze(1) == 0)[0]]}"
        )
        means = sum_per_sample / count_per_sample  # [batch_size, 1]
        input_data = input_data - means
        input_data = input_data * input_mask
        var_per_sample = (
            torch.sum(input_data**2, dim=1, keepdim=True).detach() / count_per_sample
        )  # [batch_size, 1]
        stdev = torch.sqrt(var_per_sample + 1e-9)
        input_data = input_data / stdev
        input_data = input_data * input_mask

        # recover the mask_pad_value(default:255.)
        input = input * ~(input_mask) + input_data

        return input, means, stdev

    def forward(self, input, revin):
        """Run Falcon-TST forward pass."""
        batch_size, input_len = input.shape
        # realize varied input length
        if input_len > self.seq_length:
            input = input[:, -self.seq_length :]
        elif input_len < self.seq_length:
            pad_len = self.seq_length - input_len
            input = F.pad(
                input,
                pad=(pad_len, 0),
                mode="constant",
                value=self.config.mask_pad_value,
            )
        input_len = self.seq_length

        input_mask = input != self.config.mask_pad_value

        # Step1. RevIN
        if revin:
            input, means, stdev = self.revin(input, input_mask)

        # Step2. Get rotary_pos_emb
        # rotary_pos_emb [input_len, 1, 1, kv_channels(hidden_size // num_heads)]
        rotary_pos_emb = self.rotary_pos_emb(input_len, device=input.device)

        # Step3. Do one-step inference to get mixed forecasts from multiple
        # forecast heads.
        # mixed_pred: [batch_size, max(multi_forecast_head)]
        mixed_pred = self._inference_step(
            input=input, input_mask=input_mask, rotary_pos_emb=rotary_pos_emb
        )

        # Step4. Based on the mixed forecasts, do auto-regressive inference according to
        # the step list of each forecast head
        if self.config.multi_forecast_head_type == "single":
            final_output = self._auto_regressive_single_head(
                input=input,
                input_mask=input_mask,
                FalconTST_forecast=mixed_pred,
                rotary_pos_emb=rotary_pos_emb,
            )
        else:
            raise NotImplementedError

        # Step5. RevIN
        if revin:
            final_output = final_output * (
                stdev.repeat(1, self.config.inference_length)
            )
            final_output = final_output + (
                means.repeat(1, self.config.inference_length)
            )

        return final_output.detach().float()

    def _inference_step(
        self,
        input,
        input_mask,
        rotary_pos_emb,
    ):
        """Run one inference step."""
        if self.config.do_base_forecast:
            base_forecast, _ = self.base_output_layer(input * input_mask)
        else:
            base_forecast = None

        decoder_backcast, decoder_forecast = self.decoder(
            input,  # [batch_size, seq_len]
            rotary_pos_emb,
        )

        if self.config.do_expert_forecast:
            assert decoder_forecast is not None, "decoder_forecast is None"
            if self.config.heterogeneous_moe_layer:
                decoder_forecast = self.output_layer(decoder_forecast)  # IdentityOp
            else:
                final_forecast = self.output_layer(decoder_backcast * input_mask)
                decoder_forecast = decoder_forecast + final_forecast
        else:
            # The decoder_backcast contains the mask_pad_val(default:255.)
            decoder_forecast, _ = self.output_layer(decoder_backcast * input_mask)

        if self.config.do_base_forecast:
            assert base_forecast is not None, "base_forecast is None"
            FalconTST_forecast = base_forecast + decoder_forecast
        else:
            FalconTST_forecast = decoder_forecast

        return FalconTST_forecast

    def _auto_regressive_single_head(
        self,
        input,  # [batch_size, seq_len]
        input_mask,  # [batch_size, seq_len]
        FalconTST_forecast,  # [batch_size, max(multi_forecast_head)]
        rotary_pos_emb,  # [seq_len, 1, 1, kv_channels(hidden_size // num_heads)]
        auto_regressive_strategy="from_long_to_short",
    ):
        """Run auto-regressive prediction with a single head."""
        assert self.config.multi_forecast_head_type == "single", (
            "_auto_regressive_single_head only support "
            "multi_forecast_head_type==single "
        )

        if auto_regressive_strategy == "from_long_to_short":
            # From long to short
            multi_forecast_head_list = sorted(
                self.config.multi_forecast_head_list, reverse=True
            )

            final_output = FalconTST_forecast
            while final_output.shape[1] < self.config.inference_length:
                # adaptive choose the forecast head
                remain_pred_len = self.config.inference_length - final_output.shape[1]
                for idx, head_pred_len in enumerate(multi_forecast_head_list):
                    if head_pred_len <= remain_pred_len:
                        break
                if idx == len(multi_forecast_head_list):
                    idx = len(multi_forecast_head_list) - 1
                head_pred_len = multi_forecast_head_list[idx]

                # one-step model prediction
                input = torch.cat([input, FalconTST_forecast], dim=1)[
                    :, -self.seq_length :
                ].contiguous()
                input_mask = torch.cat(
                    [
                        input_mask,
                        torch.ones(
                            FalconTST_forecast.shape,
                            dtype=input_mask.dtype,
                            device=input_mask.device,
                        ),
                    ],
                    dim=1,
                )[:, -self.seq_length :].contiguous()
                # 0: mask, 1: unmask

                FalconTST_forecast = self._inference_step(
                    input=input,
                    input_mask=input_mask,
                    rotary_pos_emb=rotary_pos_emb,
                )

                # the core idea of multi forecast head type of [single]
                FalconTST_forecast = FalconTST_forecast[:, :head_pred_len]

                final_output = torch.cat([final_output, FalconTST_forecast], dim=1)

            final_output = final_output[:, : self.config.inference_length]

        else:
            raise NotImplementedError

        assert final_output.shape[1] == self.config.inference_length
        return final_output


class FalconTSTForPrediction(FalconTSTPreTrainedModel):
    """Falcon-TST prediction model."""

    def __init__(self, config: FalconTSTConfig):
        super().__init__(config)
        self.config = config
        self.model = FalconTSTModel(self.config)
        self.post_init()

    @torch.no_grad()
    def predict(
        self,
        time_series: torch.Tensor,
        forecast_horizon: int,
        revin: bool = True,
    ) -> torch.Tensor:
        """Generate time series forecasts autoregressively.

        Args:
            time_series (torch.Tensor): Input time series data.
                Shape: [batch_size, seq_len] or
                [batch_size, seq_len, channels].
            forecast_horizon (int): The number of future time steps to predict.

        Returns
        -------
            torch.Tensor: Forecasted time series with shape
                [batch_size, forecast_horizon, channels].
        """
        self.eval()

        assert time_series.ndim == 2 or time_series.ndim == 3, (
            "Input shape must be [batch, seq_len, channel] or [batch, seq_len]"
        )
        is_multichannel = time_series.ndim == 3
        if is_multichannel:
            batch_size, seq_len, num_channels = time_series.shape
            # [B, L, C] -> [B * C, L]
            input_flat = time_series.permute(0, 2, 1).reshape(
                batch_size * num_channels, seq_len
            )
        else:
            batch_size, seq_len = time_series.shape
            num_channels = 1
            input_flat = time_series

        self.config.inference_length = forecast_horizon
        forecast_flat = self.model(input=input_flat, revin=revin)  # Shape: [B * C, H]

        if is_multichannel:
            forecast = forecast_flat.reshape(batch_size, num_channels, forecast_horizon)
            forecast = forecast.permute(0, 2, 1).contiguous()
        else:
            forecast = forecast_flat

        return forecast
