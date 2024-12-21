from typing import List

from transformers import PretrainedConfig


class TimeMoeConfig(PretrainedConfig):
    model_type = "time_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        horizon_lengths: List[int] = 1,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = None,
        hidden_act: str = "silu",
        num_experts_per_tok: int = 2,
        num_experts: int = 1,
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        use_dense: bool = False,
        rope_theta: int = 10000,
        attention_dropout: float = 0.0,
        apply_aux_loss: bool = True,
        router_aux_loss_factor: float = 0.02,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        if isinstance(horizon_lengths, int):
            horizon_lengths = [horizon_lengths]
        self.horizon_lengths = (
            horizon_lengths  # Predict horizon length for each prediction.
        )
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_dense = use_dense
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.apply_aux_loss = apply_aux_loss
        self.router_aux_loss_factor = router_aux_loss_factor

        assert (
            self.use_dense ^ self.apply_aux_loss
        ), "Both use_dense and apply_aux_loss cannot be set to True or False at the same time."

        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
