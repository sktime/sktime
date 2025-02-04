from functools import partial

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn.functional as F
    from torch import nn
    from torch.utils._pytree import tree_map

    from .distribution import DistributionOutput
    from .module.norm import RMSNorm
    from .module.position import (
        BinaryAttentionBias,
        QueryKeyProjection,
        RotaryProjection,
    )
    from .module.transformer import TransformerEncoder
    from .module.ts_embed import MultiInSizeLinear
else:

    class nn:
        class Module:
            pass

    class DistributionOutput:
        pass


if _check_soft_dependencies("huggingface-hub", severity="none"):
    from huggingface_hub import PyTorchModelHubMixin
else:
    # Create Dummy class
    class PyTorchModelHubMixin:
        def __init__(self):
            pass

        def __init_subclass__(cls, *args, **kwargs) -> None:
            """Implement dummy version of __init_subclass__."""
            pass

        pass


if _check_soft_dependencies("einops", severity="none"):
    from .module.packed_scaler import PackedNOPScaler, PackedStdScaler

from .common.torch_util import mask_fill, packed_attention_mask


def encode_distr_output(
    distr_output: DistributionOutput,
):
    """Serialize function for DistributionOutput."""

    def _encode(val):
        if not isinstance(val, DistributionOutput):
            return val

        return {
            "_target_": f"{val.__class__.__module__}.{val.__class__.__name__}",
            **tree_map(_encode, val.__dict__),
        }

    return _encode(distr_output)


def decode_distr_output(config) -> DistributionOutput:
    """Deserialize function for DistributionOutput."""
    from hydra.utils import instantiate

    return instantiate(config, _convert_="all")


class MoiraiModule(
    nn.Module,
    PyTorchModelHubMixin,
    coders={DistributionOutput: (encode_distr_output, decode_distr_output)},
):
    def __init__(
        self,
        distr_output: DistributionOutput,
        d_model: int,
        num_layers: int,
        patch_sizes: tuple[int, ...],  # tuple[int, ...] | list[int]
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_sizes = patch_sizes
        self.max_seq_len = max_seq_len
        self.scaling = scaling

        self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.in_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=None,
        )
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_param_proj(d_model, patch_sizes)

    def forward(
        self,
        target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
        patch_size,
    ):
        """
        Define the forward pass of MoiraiModule.

        This method expects processed inputs.

        1. Apply scaling to observations
        2. Project from observations to representations
        3. Replace prediction window with learnable mask
        4. Apply transformer layers
        5. Project from representations to distribution parameters
        6. Return distribution object

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :param prediction_mask: binary mask for prediction horizon,
        1 if part of the horizon, 0 otherwise
        :param patch_size: patch size for each token
        :return: predictive distribution
        """
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        reprs = self.in_proj(scaled_target, patch_size)
        masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)
        reprs = self.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )
        distr_param = self.param_proj(reprs, patch_size)
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
        return distr
