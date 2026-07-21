# ruff: noqa
"""
Part of code from time_moe.models.configuration_time_moe
https://github.com/Time-MoE
"""

from typing import List, Optional

from sktime.utils.dependencies import _safe_import

PretrainedConfig = _safe_import("transformers.PretrainedConfig")
logging = _safe_import("transformers.utils.logging")
logger = logging.get_logger(__name__)


class MIRAConfig(PretrainedConfig):
    model_type = "mira"
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
        # Time/Positional Encoding
        time_aware_rotary: bool = True,
        rope_theta: int = 10000,
        time_scale: float = 1.0,
        # Terminal ODE Module
        use_terminal_ode: bool = True,  #
        ode_func_hidden_dims: List[int] = [128, 128],
        ode_func_activation: str = "silu",
        ode_solver_method: str = "dopri5",  # e.g., 'dopri5', 'rk4'
        ode_solver_atol: float = 1e-6,
        ode_solver_rtol: float = 1e-6,
        ode_func_use_time: bool = True,  # Whether f_ODE uses relative time (s-t_N) as input
        ode_activation_threshold: float = 1.0,
        # MoE Config
        apply_aux_loss: bool = True,  # Automatically disabled if use_dense=True
        router_aux_loss_factor: float = 0.02,
        # Other
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        gradient_checkpointing_kwargs: Optional[dict] = None,  # Corrected type hint
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

        # Time/Positional Encoding
        self.rope_theta = rope_theta
        self.time_aware_rotary = time_aware_rotary
        self.time_scale = time_scale

        # Terminal ODE
        self.use_terminal_ode = use_terminal_ode
        self.ode_func_hidden_dims = ode_func_hidden_dims
        self.ode_func_activation = ode_func_activation
        self.ode_solver_method = ode_solver_method
        self.ode_solver_atol = ode_solver_atol
        self.ode_solver_rtol = ode_solver_rtol
        self.ode_func_use_time = ode_func_use_time
        self.ode_activation_threshold = ode_activation_threshold

        # MoE
        # Ensure apply_aux_loss matches MoE usage
        self.apply_aux_loss = apply_aux_loss
        self.router_aux_loss_factor = router_aux_loss_factor

        # Other
        self.attention_dropout = attention_dropout
        self.gradient_checkpointing_kwargs = (
            gradient_checkpointing_kwargs
            if gradient_checkpointing_kwargs is not None
            else {"use_reentrant": True}
        )

        assert self.use_dense ^ self.apply_aux_loss, (
            "Both use_dense and apply_aux_loss cannot be set to True or False at the same time."
        )

        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
