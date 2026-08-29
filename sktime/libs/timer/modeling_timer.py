"""PyTorch implementation of the Timer forecasting model."""

from sktime.utils.dependencies import _safe_import

from .configuration_timer import TimerConfig
from .ts_generation_mixin import TSGenerationMixin

torch = _safe_import("torch")
F = _safe_import("torch.nn.functional")
nn = _safe_import("torch.nn")
Cache = _safe_import("transformers.Cache")
DynamicCache = _safe_import("transformers.DynamicCache")
PreTrainedModel = _safe_import("transformers.PreTrainedModel")
ACT2FN = _safe_import("transformers.activations.ACT2FN")
_prepare_4d_causal_attention_mask = _safe_import(
    "transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask"
)
MoeCausalLMOutputWithPast = _safe_import(
    "transformers.modeling_outputs.MoeCausalLMOutputWithPast"
)
MoeModelOutputWithPast = _safe_import(
    "transformers.modeling_outputs.MoeModelOutputWithPast"
)


def rotate_half(x):
    """Rotate half of the hidden dimensions for rotary embeddings."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Apply rotary position embeddings to query and key states."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimerPatchEmbedding(nn.Module):
    """Embed fixed-length time-series patches."""

    def __init__(self, config: TimerConfig):
        super().__init__()
        self.input_token_len = config.input_token_len
        self.emb = nn.Linear(config.input_token_len, config.hidden_size, bias=False)

    def forward(self, hidden_state: torch.Tensor):
        """Return patch embeddings for the input time series."""
        hidden_state = hidden_state.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len
        )
        return self.emb(hidden_state)


class TimerPointEmbedding(nn.Module):
    """Embed point tokens with a gated projection."""

    def __init__(self, config: TimerConfig):
        super().__init__()
        self.emb_layer = nn.Linear(
            config.input_token_len, config.hidden_size, bias=False
        )
        self.gate_layer = nn.Linear(
            config.input_token_len, config.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """Return gated point embeddings."""
        emb = self.act_fn(self.gate_layer(x)) * self.emb_layer(x)
        return emb


class TimeMoeRotaryEmbedding(torch.nn.Module):
    """Rotary positional embedding cache for Timer attention."""

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

        # Build here to make `torch.jit.trace` work.
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
        # Uses a different permutation from the paper for the same calculation.
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """Return cached rotary embedding tensors for the requested length."""
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class TimerAttention(nn.Module):
    """Multi-head causal self-attention used by Timer."""

    def __init__(self, config: TimerConfig, layer_idx: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = TimeMoeRotaryEmbedding(
            self.head_dim, max_position_embeddings=config.max_position_embeddings
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Apply self-attention to hidden states."""
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout_p=self.attention_dropout,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class TimerMLP(nn.Module):
    """Feed-forward network used in Timer decoder layers."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        """Apply the gated feed-forward network."""
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


class TimerDecoderLayer(nn.Module):
    """Single Timer decoder block."""

    def __init__(self, config: TimerConfig, layer_idx: int):
        super().__init__()
        self.self_attn = TimerAttention(config, layer_idx)

        self.ffn_layer = TimerMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.norm1 = torch.nn.LayerNorm(config.hidden_size)
        self.norm2 = torch.nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor | None,
        torch.FloatTensor | None,
    ]:
        """Run one decoder block."""
        residual = hidden_states

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
        hidden_states = self.norm1(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_layer(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm2(hidden_states)

        if not output_attentions:
            self_attn_weights = None

        if not use_cache:
            present_key_value = None
        return hidden_states, self_attn_weights, present_key_value


class TimerPreTrainedModel(PreTrainedModel):
    """Base class for Timer pretrained models."""

    config_class = TimerConfig
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


class TimerModel(TimerPreTrainedModel):
    """Timer decoder model without prediction heads."""

    def __init__(self, config: TimerConfig):
        super().__init__(config)
        self.embed_layer = TimerPatchEmbedding(config)
        self.layers = nn.ModuleList(
            [
                TimerDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | MoeModelOutputWithPast:
        """Run the Timer decoder model."""
        # input_ids is the input of time series, its shape is [batch_size, seq_len]
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
                "You cannot specify both decoder_input_ids and "
                "decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_layer(input_ids)
            seq_length = inputs_embeds.shape[1]

        if self.gradient_checkpointing and self.training:
            if use_cache:
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
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class TimerForPrediction(TimerPreTrainedModel, TSGenerationMixin):
    """Timer model with forecasting heads."""

    def __init__(self, config: TimerConfig):
        super().__init__(config)
        self.config = config
        self.model = TimerModel(self.config)
        lm_head_list = []
        self.output_token_len_map = {}
        for i, output_token_len in enumerate(self.config.output_token_lens):
            lm_head_list.append(
                nn.Linear(self.config.hidden_size, output_token_len, bias=False)
            )
            self.output_token_len_map[output_token_len] = i
        self.lm_heads = nn.ModuleList(lm_head_list)
        self.loss_function = torch.nn.MSELoss(reduction="none")
        self.post_init()

    def set_decoder(self, decoder):
        """Set the underlying decoder model."""
        self.model = decoder

    def get_decoder(self):
        """Return the underlying decoder model."""
        return self.model

    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.FloatTensor | None = None,
        loss_masks: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        max_output_length: int | None = None,
        revin: bool | None = False,
    ) -> tuple | MoeCausalLMOutputWithPast:
        """Run Timer forecasting or training loss computation."""
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
            mean, std = (
                input_ids.mean(dim=-1, keepdim=True),
                input_ids.std(dim=-1, keepdim=True),
            )
            input_ids = (input_ids - mean) / std
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

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        predictions = None

        loss = None
        if labels is not None:
            ar_loss = 0.0
            for lm_head, output_token_len in zip(
                self.lm_heads, self.config.output_token_lens
            ):
                one_predictions = lm_head(hidden_states)
                one_loss = self.calc_ar_loss(
                    one_predictions, labels, loss_masks, output_token_len
                )
                ar_loss += one_loss
                if predictions is None:
                    predictions = one_predictions
            loss = ar_loss / len(self.config.output_token_lens)
        else:
            if max_output_length is None:
                output_token_len = self.config.output_token_lens[0]
                max_output_length = output_token_len
            else:
                output_token_len = self.config.output_token_lens[0]
                for h in self.config.output_token_lens[1:]:
                    if h > max_output_length:
                        break
                    else:
                        output_token_len = h
            lm_head = self.lm_heads[self.output_token_len_map[output_token_len]]
            predictions = lm_head(hidden_states)[:, -1, :]
            if output_token_len > max_output_length:
                predictions = predictions[:, :max_output_length]
            if revin:
                predictions = predictions * std + mean
        if not return_dict:
            output = (predictions,) + outputs[1:]
            return (loss) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=predictions,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def calc_ar_loss(self, predictions, labels, loss_masks, output_token_len):
        """Calculate autoregressive forecasting loss."""
        seq_len = predictions.shape[1] * self.config.input_token_len
        labels = labels[:, : seq_len - self.config.input_token_len + output_token_len]
        shift_labels = labels.unfold(
            dimension=-1, size=output_token_len, step=self.config.input_token_len
        )

        # Calculate loss with mask
        losses = self.loss_function(predictions, shift_labels).mean(dim=-1)
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
        revin=True,
        **kwargs,
    ):
        """Prepare model inputs for autoregressive generation."""
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
            # 1 - If attention_mask is longer than input_ids, some inputs are
            # passed exclusively as part of the cache, e.g. input_embeds.
            if attention_mask is not None and attention_mask.shape[1] > (
                input_ids.shape[1] // self.config.input_token_len
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If past_length is smaller than input_ids, then input_ids
            # holds all input tokens. Discard the cached prefix.
            elif past_length < (input_ids.shape[1] // self.config.input_token_len):
                input_ids = input_ids[:, past_length * self.config.input_token_len :]
            # 3 - Otherwise, assume input_ids has only unprocessed tokens.

            # Crop the input attention mask before exceeding max cache length.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + (input_ids.shape[1] // self.config.input_token_len)
                > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[
                    :, -(input_ids.shape[1] // self.config.input_token_len) :
                ]

        # Use `inputs_embeds` only in the first generation step.
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
            }
        )
        return model_inputs
