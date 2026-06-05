from typing import List
from transformers import PretrainedConfig


class TimerConfig(PretrainedConfig):
    model_type = "timer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        input_token_len: int = 1,
        hidden_size: int = 1024,
        intermediate_size: int = 2048,
        output_token_lens: List[int] = [1, 8, 32, 64],
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        hidden_act: str = "silu",
        use_cache: bool = True,
        rope_theta: int = 10000,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 10000,
        **kwargs,
    ):
        self.input_token_len = input_token_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.output_token_lens = output_token_lens
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings

        super().__init__(
            **kwargs,
        )
