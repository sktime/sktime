"""
Configuration class for FalconTST model.

This module defines the configuration for FalconTST, a large-scale time series
foundation model that utilizes Mixture of Experts (MoE) architecture with
multiple patch tokenizers.
"""

from sktime.utils.dependencies import _safe_import

PretrainedConfig = _safe_import("transformers.PretrainedConfig")


class FalconTSTConfig(PretrainedConfig):
    """
    Configuration class for FalconTST model.

    FalconTST is a time series foundation model that uses Mixture of Experts
    architecture with multiple patch tokenizers for efficient time series
    forecasting.

    This configuration inherits from [`PretrainedConfig`] and can be used to
    control the model output. Read the documentation from [`PretrainedConfig`]
    for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        ffn_hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the feed-forward networks in the transformer layers.
        seq_length (`int`, *optional*, defaults to 2880):
            Maximum sequence length that the model can handle.
        add_bias_linear (`bool`, *optional*, defaults to `False`):
            Whether to add bias in linear layers.
        rope_theta (`int`, *optional*, defaults to 10000):
            The base period of the RoPE embeddings.
        num_hidden_layers (`int`, *optional*, defaults to 3):
            Number of hidden layers in the transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the
            transformer encoder.
        mask_pad_value (`float`, *optional*, defaults to 255.0):
            Value used for padding/masking in input sequences.
        expert_num_layers (`int`, *optional*, defaults to 4):
            Number of transformer layers within each expert.
        shared_patch_size (`int`, *optional*, defaults to 64):
            Size of patches for the shared expert.
        patch_size_list (`List[int]`, *optional*, defaults to [96, 64, 48, 24]):
            List of patch sizes for different experts.
        multi_forecast_head_list (`List[int]`, *optional*, defaults to [24, 96, 336]):
            List of forecast lengths for multi-head prediction.
        is_revin (`bool`, *optional*, defaults to `True`):
            Whether to use RevIN (Reversible Instance Normalization).
        params_dtype (`str`, *optional*, defaults to "bfloat16"):
            Data type for model parameters.
        use_cpu_initialization (`bool`, *optional*, defaults to `False`):
            Whether to initialize model parameters on CPU.
        rotary_interleaved (`bool`, *optional*, defaults to `False`):
            Whether to use interleaved rotary position embeddings.
        do_expert_forecast (`bool`, *optional*, defaults to `True`):
            Whether experts perform forecasting.
        residual_backcast (`bool`, *optional*, defaults to `True`):
            Whether to use residual connections for backcast.
        do_base_forecast (`bool`, *optional*, defaults to `False`):
            Whether to use base forecasting.
        heterogeneous_moe_layer (`bool`, *optional*, defaults to `True`):
            Whether to use heterogeneous MoE layers.
        test_data_seq_len (`int`, *optional*, defaults to 2880):
            Sequence length for test data.
        test_data_test_len (`int`, *optional*, defaults to 720):
            Test length for test data.
        autoregressive_step_list (`List[int]`, *optional*, defaults to [2, 4, 1]):
            List of autoregressive steps for different forecast heads.
        multi_forecast_head_type (`str`, *optional*, defaults to "single"):
            Type of multi-forecast head aggregation.
        num_experts (`int`, *optional*, defaults to 4):
            Number of experts in the MoE layer.
        moe_router_topk (`int`, *optional*, defaults to 2):
            Number of top experts to route each token to.
        moe_ffn_hidden_size (`int`, *optional*, defaults to 4096):
            Hidden size for MoE feed-forward networks.
        moe_shared_expert_intermediate_size (`int`, *optional*, defaults to 4096):
            Intermediate size for shared experts.
        init_method_std (`float`, *optional*, defaults to 0.06):
            Standard deviation for weight initialization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Range for weight initialization.
        moe_router_enable_expert_bias (`bool`, *optional*, defaults to `False`):
            Whether to enable expert bias in routing.
        moe_expert_final_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization at the end of each expert.
        transformer_input_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization to transformer inputs.
        moe_router_pre_softmax (`bool`, *optional*, defaults to `True`):
            Whether to apply softmax before routing.
        q_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization to query vectors.
        k_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization to key vectors.
        moe_router_score_function (`str`, *optional*, defaults to "softmax"):
            Score function for MoE routing ("softmax" or "sigmoid").
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embeddings.
    """

    model_type = "FalconTST"

    def __init__(
        self,
        # model configs
        add_bias_linear: bool = False,
        num_hidden_layers: int = 3,
        hidden_size: int = 1024,
        ffn_hidden_size: int = 4096,
        num_attention_heads: int = 16,
        seq_length: int = 2880,
        mask_pad_value: float = 255.0,
        is_revin: bool = True,
        shared_patch_size: int = 32,
        patch_size_list: list[int] | None = None,
        residual_backcast: bool = True,
        do_base_forecast: bool = False,
        do_expert_forecast: bool = True,
        heterogeneous_moe_layer: bool = False,
        expert_num_layers: int = 4,
        multi_forecast_head_list: list[int] | None = None,
        multi_forecast_head_type: str = "single",
        rope_theta: int = 1000000,
        rotary_interleaved: bool = False,
        block_input_layernorm: bool = True,
        transformer_input_layernorm: bool = True,
        # moe configs
        num_experts: int = 4,
        moe_router_topk: int = 2,
        moe_router_pre_softmax: bool = True,
        moe_router_score_function: str = "softmax",
        moe_ffn_hidden_size: int = 4096,
        moe_shared_expert_intermediate_size: int = 4096,
        moe_router_enable_expert_bias: bool = False,
        moe_expert_final_layernorm: bool = True,
        # initial configs
        use_cpu_initialization: bool = False,
        init_method_std: float = 0.06,
        initializer_range: float = 0.02,
        # test configs
        test_data_seq_len: int = 2880,
        test_data_test_len: int = 720,
        autoregressive_step_list: list[int] | None = None,
        **kwargs,
    ):
        """Initialize FalconTST configuration."""
        # model configs
        self.add_bias_linear = add_bias_linear
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_attention_heads = num_attention_heads
        self.kv_channels = self.hidden_size // self.num_attention_heads
        self.seq_length = seq_length
        self.mask_pad_value = mask_pad_value
        self.is_revin = is_revin
        self.shared_patch_size = shared_patch_size
        if patch_size_list is None:
            patch_size_list = [96, 64, 48, 24]
        self.patch_size_list = patch_size_list
        self.residual_backcast = residual_backcast
        self.do_base_forecast = do_base_forecast
        self.do_expert_forecast = do_expert_forecast
        self.heterogeneous_moe_layer = heterogeneous_moe_layer
        self.expert_num_layers = expert_num_layers
        if multi_forecast_head_list is None:
            multi_forecast_head_list = [24, 96, 336]
        self.multi_forecast_head_list = multi_forecast_head_list
        self.pred_length = max(self.multi_forecast_head_list)
        self.multi_forecast_head_type = multi_forecast_head_type
        self.rotary_base = rope_theta
        self.rotary_interleaved = rotary_interleaved
        self.block_input_layernorm = block_input_layernorm
        self.transformer_input_layernorm = transformer_input_layernorm

        # moe configs
        self.num_moe_experts = num_experts
        self.moe_router_topk = moe_router_topk
        self.moe_router_input_size = self.seq_length
        self.moe_router_pre_softmax = moe_router_pre_softmax
        self.moe_router_score_function = moe_router_score_function
        self.moe_ffn_hidden_size = moe_ffn_hidden_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.moe_expert_final_layernorm = moe_expert_final_layernorm

        # initial configs
        self.use_cpu_initialization = use_cpu_initialization
        self.init_method_std = init_method_std
        self.initializer_range = initializer_range

        # test configs
        self.test_data_seq_len = test_data_seq_len
        self.inference_length = test_data_test_len
        if autoregressive_step_list is None:
            autoregressive_step_list = [2, 4, 1]
        self.autoregressive_step_list = autoregressive_step_list

        self.use_cache = True

        super().__init__(
            **kwargs,
        )
