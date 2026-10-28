"""ConvTimeNet neural network architecture in PyTorch."""

__authors__ = ["Tanuj-Taneja1"]
__all__ = ["ConvTimeNetNetworkTorch"]

from collections.abc import Callable

from sktime.networks.convtimenet._convtimenet_backbone import ConvTimeNet_backbone
from sktime.networks.convtimenet._dlutils import DeformablePatch
from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class ConvTimeNetNetworkTorch(NNModule):
    """Establish the network structure for ConvTimeNet in PyTorch.

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    enc_in : int
        Number of input channels, i.e., variables of the time series.
    seq_len : int
        Length of the input series.
    n_classes : int
        Number of outputs, i.e., the number of classes. Use ``1`` for regression.
    d_model : int, default=64
        Hidden dimension size for model processing.
    patch_size : int, default=4
        Size of the patches the series is split into.
    patch_stride : int, default=2
        Stride between consecutive patches.
    dropout : float, default=0.0
        Dropout rate applied in the encoder blocks.
    d_ff : int, default=128
        Dimension of the position-wise feed-forward networks.
    dw_ks : list of int or None, default=None
        Depthwise convolution kernel size of each encoder block. The number of
        encoder blocks equals the length of this list. If None, defaults to
        ``[7, 13, 19]``.
    activation : Callable or None, default=None
        Activation applied to the output layer. If None, the network returns
        raw outputs, i.e., logits.
    activation_hidden : Callable or None, default=None
        Activation applied in the hidden layers. If None, no activation is
        applied in the hidden layers.
    device : str, default="cpu"
        Device the network is placed on.

    References
    ----------
    .. [1] Cheng, M., Yang, J., Pan, T., Liu, Q., & Li, Z. (2024). ConvTimeNet: A deep
        hierarchical fully convolutional model for multivariate time series analysis.
        arXiv preprint arXiv:2403.01493. https://arxiv.org/abs/2403.01493
    """

    _tags = {
        "authors": ["Tanuj-Taneja1"],
        "maintainers": ["Tanuj-Taneja1"],
        "python_dependencies": ["torch"],
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        n_classes: int,
        d_model: int = 64,
        patch_size: int = 4,
        patch_stride: int = 2,
        dropout: float = 0.0,
        d_ff: int = 128,
        dw_ks: list[int] | None = None,
        activation: Callable | None = None,
        activation_hidden: Callable | None = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.enc_in = enc_in
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.dropout = dropout
        self.d_ff = d_ff
        self.dw_ks = [7, 13, 19] if dw_ks is None else dw_ks
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.device = device

        # DePatch Embedding
        self.depatchEmbedding = DeformablePatch(
            seq_len=self.seq_len,
            patch_size=self.patch_size,
            stride=self.patch_stride,
            in_feats=self.enc_in,
            out_feats=self.d_model,
            activation_hidden=self.activation_hidden,
        )

        # ConvTimeNet Backbone
        self.main_net = ConvTimeNet_backbone(
            c_in=self.d_model,
            c_out=self.n_classes,
            seq_len=self.depatchEmbedding.new_len,
            n_layers=len(self.dw_ks),
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation_hidden=self.activation_hidden,
            pooling_tp="max",
            fc_dropout=0.0,
            enable_res_param=True,
            dw_ks=self.dw_ks,
            norm="batch",
            use_embed=False,
            re_param=True,
            device=self.device,
        )

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_len, enc_in)
            Input tensor containing the time series data.

        Returns
        -------
        torch.Tensor of shape (batch_size, n_classes)
            Class scores. Raw outputs, i.e., logits, unless ``activation``
            is passed.
        """
        out_patch = self.depatchEmbedding(X)  # [bs, features]
        output = self.main_net(out_patch.permute(0, 2, 1))
        if self.activation is not None:
            output = self.activation(output)
        return output
