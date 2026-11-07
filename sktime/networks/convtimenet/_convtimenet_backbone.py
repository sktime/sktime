"""Backbone of the ConvTimeNet classification network (PyTorch)."""

__all__ = ["ConvTimeNet_backbone"]
__authors__ = ["Tanuj-Taneja1"]

import copy

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class SublayerConnection(NNModule):
    """Residual connection with optional learnable residual weight.

    Parameters
    ----------
    enable_res_parameter : bool
        Whether to scale the residual branch by a learnable parameter.
    dropout : float, default=0.1
        Dropout rate applied to the residual branch.
    """

    def __init__(self, enable_res_parameter, dropout=0.1):
        super().__init__()
        nnDropout = _safe_import("torch.nn.Dropout")
        self.dropout = nnDropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            nnParameter = _safe_import("torch.nn.Parameter")
            torch_tensor = _safe_import("torch.tensor")
            self.a = nnParameter(torch_tensor(0.5))

    def forward(self, x, out_x):
        """Add the residual branch to the skip connection.

        Parameters
        ----------
        x : torch.Tensor
            Input of the sublayer, i.e., the skip connection.
        out_x : torch.Tensor
            Output of the sublayer, i.e., the residual branch.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``.
        """
        if not self.enable:
            return x + self.dropout(out_x)
        else:
            return x + self.dropout(self.a * out_x)


class _ConvEncoderLayer(NNModule):
    """Single depthwise convolutional encoder block of ConvTimeNet.

    Parameters
    ----------
    kernel_size : int
        Kernel size of the depthwise convolution.
    d_model : int
        Hidden dimension of the block.
    d_ff : int, default=256
        Dimension of the position-wise feed-forward network.
    dropout : float, default=0.1
        Dropout rate applied in the residual and feed-forward branches.
    activation_hidden : torch.nn.Module or None, default=None
        Activation applied inside the block. If None, no activation is applied.
    enable_res_param : bool, default=True
        Whether to scale the residual branches by a learnable parameter.
    norm : str, default="batch"
        Normalization applied in the block, ``"batch"`` or ``"layer"``.
    small_ks : int, default=3
        Kernel size of the small depthwise convolution used for re-parametrization.
    re_param : bool, default=True
        Whether to use the large kernel re-parametrization mechanism.
    device : str, default="cpu"
        Device the block is placed on.
    """

    def __init__(
        self,
        kernel_size,
        d_model,
        d_ff=256,
        dropout=0.1,
        activation_hidden=None,
        enable_res_param=True,
        norm="batch",
        small_ks=3,
        re_param=True,
        device="cpu",
    ):
        super().__init__()

        nnSequential = _safe_import("torch.nn.Sequential")
        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnDropout = _safe_import("torch.nn.Dropout")
        nnIdentity = _safe_import("torch.nn.Identity")
        nnLayerNorm = _safe_import("torch.nn.LayerNorm")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")

        self.norm_tp = norm
        self.re_param = re_param

        # DeepWise Conv. Add & Norm
        if self.re_param:
            self.large_ks = kernel_size
            self.small_ks = small_ks
            self.DW_conv_large = nnConv1d(
                d_model,
                d_model,
                self.large_ks,
                stride=1,
                padding="same",
                groups=d_model,
            )
            self.DW_conv_small = nnConv1d(
                d_model,
                d_model,
                self.small_ks,
                stride=1,
                padding="same",
                groups=d_model,
            )
            self.DW_infer = nnConv1d(
                d_model,
                d_model,
                self.large_ks,
                stride=1,
                padding="same",
                groups=d_model,
            )
        else:
            self.DW_conv = nnConv1d(
                d_model, d_model, kernel_size, stride=1, padding="same", groups=d_model
            )

        act = nnIdentity() if activation_hidden is None else activation_hidden
        self.dw_act = act

        self.sublayerconnect1 = SublayerConnection(enable_res_param, dropout)
        self.dw_norm = (
            nnBatchNorm1d(d_model) if norm == "batch" else nnLayerNorm(d_model)
        )

        # Position-wise Feed-Forward
        self.ff = nnSequential(
            nnConv1d(d_model, d_ff, 1, 1),
            act,
            nnDropout(dropout),
            nnConv1d(d_ff, d_model, 1, 1),
        )

        # Add & Norm
        self.sublayerconnect2 = SublayerConnection(enable_res_param, dropout)
        self.norm_ffn = (
            nnBatchNorm1d(d_model) if norm == "batch" else nnLayerNorm(d_model)
        )

    def _get_merge_param(self):
        nnParameter = _safe_import("torch.nn.Parameter")
        pad = _safe_import("torch.nn.functional.pad")

        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad

        module_output = copy.deepcopy(self.DW_conv_large)

        module_output.weight = nnParameter(
            module_output.weight
            + pad(self.DW_conv_small.weight, (left_pad, right_pad), value=0)
        )

        module_output.bias = nnParameter(module_output.bias + self.DW_conv_small.bias)

        self.DW_infer = module_output

    def forward(self, src):
        """Run one encoder block.

        Parameters
        ----------
        src : torch.Tensor of shape (batch_size, d_model, seq_len)
            Input tensor of the block.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model, seq_len)
            Output tensor of the block.
        """
        ## Deep-wise Conv Layer
        if not self.re_param:
            src = self.DW_conv(src)
        else:
            if self.training:  # training phase
                large_out, small_out = self.DW_conv_large(src), self.DW_conv_small(src)
                src = self.sublayerconnect1(src, self.dw_act(large_out + small_out))
            else:  # testing phase
                self._get_merge_param()
                merge_out = self.DW_infer(src)
                src = self.sublayerconnect1(src, self.dw_act(merge_out))

        src = src.permute(0, 2, 1) if self.norm_tp != "batch" else src
        src = self.dw_norm(src)
        src = src.permute(0, 2, 1) if self.norm_tp != "batch" else src

        ## Position-wise Conv Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm

        # Add: residual connection with residual dropout
        src2 = self.sublayerconnect2(src, src2)

        # Norm: batchnorm or layernorm
        src2 = src2.permute(0, 2, 1) if self.norm_tp != "batch" else src2
        src2 = self.norm_ffn(src2)
        src2 = src2.permute(0, 2, 1) if self.norm_tp != "batch" else src2

        return src2


class _ConvEncoder(NNModule):
    """Stack of depthwise convolutional encoder blocks.

    Parameters
    ----------
    d_model : int
        Hidden dimension of the blocks.
    d_ff : int
        Dimension of the position-wise feed-forward networks.
    kernel_size : list of int or None, default=None
        Depthwise convolution kernel size of each block. If None, defaults to
        ``[19, 19, 29, 29, 37, 37]``.
    dropout : float, default=0.1
        Dropout rate applied in the blocks.
    activation_hidden : torch.nn.Module or None, default=None
        Activation applied inside the blocks. If None, no activation is applied.
    n_layers : int, default=3
        Number of encoder blocks.
    enable_res_param : bool, default=False
        Whether to scale the residual branches by a learnable parameter.
    norm : str, default="batch"
        Normalization applied in the blocks, ``"batch"`` or ``"layer"``.
    re_param : bool, default=False
        Whether to use the large kernel re-parametrization mechanism.
    device : str, default="cpu"
        Device the encoder is placed on.
    """

    def __init__(
        self,
        d_model,
        d_ff,
        kernel_size=None,
        dropout=0.1,
        activation_hidden=None,
        n_layers=3,
        enable_res_param=False,
        norm="batch",
        re_param=False,
        device="cpu",
    ):
        super().__init__()
        nnModuleList = _safe_import("torch.nn.ModuleList")

        if kernel_size is None:
            kernel_size = [19, 19, 29, 29, 37, 37]
        self.layers = nnModuleList(
            [
                _ConvEncoderLayer(
                    kernel_size[i],
                    d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation_hidden=activation_hidden,
                    enable_res_param=enable_res_param,
                    norm=norm,
                    re_param=re_param,
                    device=device,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, src):
        """Run the stack of encoder blocks.

        Parameters
        ----------
        src : torch.Tensor of shape (batch_size, d_model, seq_len)
            Input tensor of the encoder.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model, seq_len)
            Output tensor of the encoder.
        """
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


class ConvTimeNet_backbone(NNModule):
    """ConvTimeNet backbone, a hierarchical fully convolutional encoder.

    The input must be standardized by variable, based on the entire training set,
    as described in the reference implementation.

    Parameters
    ----------
    c_in : int
        Number of input channels of the encoder.
    c_out : int
        Number of outputs, i.e., the number of classes.
    seq_len : int
        Length of the (patched) input sequence.
    n_layers : int, default=3
        Number of encoder blocks. Must match the length of ``dw_ks``.
    d_model : int, default=128
        Hidden dimension of the encoder blocks.
    d_ff : int, default=256
        Dimension of the position-wise feed-forward networks.
    dropout : float, default=0.1
        Dropout rate applied in the encoder blocks.
    activation_hidden : torch.nn.Module or None, default=None
        Activation applied inside the encoder blocks and the head.
        If None, no activation is applied.
    pooling_tp : str, default="max"
        Pooling used in the head, one of ``"max"``, ``"mean"`` or ``"cat"``.
    fc_dropout : float, default=0.0
        Dropout rate applied in the head, only used if ``pooling_tp="cat"``.
    enable_res_param : bool, default=False
        Whether to scale the residual branches by a learnable parameter.
    dw_ks : list of int or None, default=None
        Depthwise convolution kernel size of each block. If None, defaults to
        ``[7, 13, 19]``.
    norm : str, default="batch"
        Normalization applied in the encoder blocks, ``"batch"`` or ``"layer"``.
    use_embed : bool, default=True
        Whether to linearly project the input to ``d_model`` before the encoder.
    re_param : bool, default=False
        Whether to use the large kernel re-parametrization mechanism.
    device : str, default="cpu"
        Device the backbone is placed on.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        n_layers: int = 3,
        d_model: int = 128,
        d_ff: int = 256,
        dropout=0.1,
        activation_hidden=None,
        pooling_tp="max",
        fc_dropout: float = 0.0,
        enable_res_param=False,
        dw_ks=None,
        norm="batch",
        use_embed=True,
        re_param=False,
        device: str = "cpu",
    ):
        super().__init__()
        nnLinear = _safe_import("torch.nn.Linear")
        nnDropout = _safe_import("torch.nn.Dropout")
        nnFlatten = _safe_import("torch.nn.Flatten")

        if dw_ks is None:
            dw_ks = [7, 13, 19]
        if n_layers != len(dw_ks):
            raise ValueError(
                "`dw_ks` should match the `n_layers` of the ConvTimeNet backbone. "
                f"Found n_layers={n_layers} and dw_ks of length {len(dw_ks)}."
            )

        self.c_out, self.seq_len = c_out, seq_len

        # Input Embedding
        self.use_embed = use_embed
        self.W_P = nnLinear(c_in, d_model)

        # Residual dropout
        self.dropout = nnDropout(dropout)

        # Encoder
        self.encoder = _ConvEncoder(
            d_model,
            d_ff,
            kernel_size=dw_ks,
            dropout=dropout,
            activation_hidden=activation_hidden,
            n_layers=n_layers,
            enable_res_param=enable_res_param,
            norm=norm,
            re_param=re_param,
            device=device,
        )

        self.flatten = nnFlatten()

        # Head
        self.head_nf = seq_len * d_model if pooling_tp == "cat" else d_model
        self.head = self.create_head(
            self.head_nf,
            c_out,
            activation_hidden=activation_hidden,
            pooling_tp=pooling_tp,
            fc_dropout=fc_dropout,
        )

    def create_head(
        self,
        nf,
        c_out,
        activation_hidden=None,
        pooling_tp="max",
        fc_dropout=0.0,
        **kwargs,
    ):
        """Create the classification head of the backbone.

        Parameters
        ----------
        nf : int
            Number of input features of the final linear layer.
        c_out : int
            Number of outputs, i.e., the number of classes.
        activation_hidden : torch.nn.Module or None, default=None
            Activation applied before flattening, only used if
            ``pooling_tp="cat"``. If None, no activation is applied.
        pooling_tp : str, default="max"
            Pooling used in the head, one of ``"max"``, ``"mean"`` or ``"cat"``.
        fc_dropout : float, default=0.0
            Dropout rate applied in the head, only used if ``pooling_tp="cat"``.

        Returns
        -------
        torch.nn.Sequential
            The classification head.
        """
        nnSequential = _safe_import("torch.nn.Sequential")
        nnLinear = _safe_import("torch.nn.Linear")
        nnDropout = _safe_import("torch.nn.Dropout")
        nnIdentity = _safe_import("torch.nn.Identity")
        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")
        nnAdaptiveMaxPool1d = _safe_import("torch.nn.AdaptiveMaxPool1d")

        layers = []
        if pooling_tp == "cat":
            act = nnIdentity() if activation_hidden is None else activation_hidden
            layers = [act, self.flatten]
            if fc_dropout:
                layers += [nnDropout(fc_dropout)]
        elif pooling_tp == "mean":
            layers = [nnAdaptiveAvgPool1d(1), self.flatten]
        elif pooling_tp == "max":
            layers = [nnAdaptiveMaxPool1d(1), self.flatten]

        layers += [nnLinear(nf, c_out)]

        # could just be used in classifying task
        return nnSequential(*layers)

    def forward(self, x):
        """Run the backbone.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, c_in, seq_len)
            Input tensor of the backbone.

        Returns
        -------
        torch.Tensor of shape (batch_size, c_out)
            Raw outputs, i.e., logits, of the classification head.
        """
        # Input encoding
        u = x
        if self.use_embed:
            u = self.W_P(x.transpose(2, 1))

        # Encoder
        z = self.encoder(u.transpose(2, 1).contiguous())  # z: [bs x d_model x q_len]

        # Classification/ Regression head
        return self.head(z)  # output: [bs x c_out]
