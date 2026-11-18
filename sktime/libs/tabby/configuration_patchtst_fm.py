# Copyright contributors to the TSFM project
#
# ruff: noqa
"""PatchTST-FM model configuration"""

from sktime.utils.dependencies import _safe_import

PretrainedConfig = _safe_import("transformers.configuration_utils.PretrainedConfig")
logging = _safe_import("transformers.utils.logging")


logger = logging.get_logger(__name__)

PATCHTSTFM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class PatchTSTFMConfig(PretrainedConfig):
    model_type = "patchtst_fm"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layer",
    }

    # has_no_defaults_at_init = True
    def __init__(
        self,
        context_length: int = 8192,
        prediction_length: int = 64,
        d_patch: int = 16,
        d_model: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        norm_first: bool = True,
        pretrain_mask_ratio: float = 0.4,
        pretrain_mask_cont: int = 8,
        num_quantile: int = 99,
        dropout: float = 0.1,
        **kwargs,
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_patch = d_patch
        self.n_patch = int(context_length // d_patch)
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.norm_first = norm_first
        self.pretrain_mask_ratio = pretrain_mask_ratio
        self.pretrain_mask_cont = pretrain_mask_cont
        self.num_quantile = num_quantile
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        self.dropout = float(dropout)
        if num_quantile % 9 == 0:
            quantiles = [
                i / (self.num_quantile + 1) for i in range(1, self.num_quantile + 1)
            ]
        else:
            quantiles = [
                i / (self.num_quantile - 1) for i in range(1, self.num_quantile - 1)
            ]
            quantiles = [0.01] + quantiles + [0.99]
        self.quantile_levels = quantiles
        # self.quantiles = quantiles
        super().__init__(**kwargs)
