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

"""Configuration objects for TimerS1 forecasting models."""

from sktime.utils.dependencies import _safe_import

PretrainedConfig = _safe_import("transformers.PretrainedConfig")


class TimerS1Config(PretrainedConfig):
    """Configuration for TimerS1 model architecture and generation parameters."""

    model_type = "Timer-S1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        input_token_len: int = 16,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        output_token_lens: list[int] = [16],
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        hidden_act: str = "silu",
        use_cache: bool = True,
        rope_theta: int = 10000,
        dropout_rate: float = 0.1,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 12800,
        quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        num_experts: int = 32,
        num_experts_per_token: int = 2,
        # MTP configuration
        num_mtp_tokens: int = 16,
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
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.quantiles = quantiles
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        # MTP configuration
        self.num_mtp_tokens = num_mtp_tokens
        super().__init__(**kwargs)
