"""Utility layers for the ConvTimeNet classification network (PyTorch)."""

__authors__ = ["Tanuj-Taneja1"]
__all__ = [
    "BoxCoder",
    "DeformablePatch",
    "OffsetPredictor",
    "PositionalEncoding",
    "SimplePatch",
    "weights_init",
]

import math

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")

_TORCH_OPS = {}


def _torch_op(import_path):
    """Lazily import and cache a torch callable used in a forward pass.

    Parameters
    ----------
    import_path : str
        Path of the torch callable, e.g. ``"torch.nn.functional.pad"``.

    Returns
    -------
    callable
        The imported torch callable, or a placeholder if ``torch`` is not present.
    """
    if import_path not in _TORCH_OPS:
        _TORCH_OPS[import_path] = _safe_import(import_path)
    return _TORCH_OPS[import_path]


def weights_init(mod):
    """Initialize the weights of a convolution, batch norm or linear module in place.

    Parameters
    ----------
    mod : torch.nn.Module
        Module whose weights need initialization.
    """
    classname = mod.__class__.__name__
    if classname.find("Conv") != -1:
        xavier_normal_ = _safe_import("torch.nn.init.xavier_normal_")
        xavier_normal_(mod.weight.data)
    elif classname.find("BatchNorm") != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        xavier_uniform_ = _safe_import("torch.nn.init.xavier_uniform_")
        xavier_uniform_(mod.weight)
        if mod.bias is not None:
            mod.bias.data.fill_(0.01)


class PositionalEncoding(NNModule):
    """Static sinusoidal positional encoding.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    dropout : float, default=0.1
        Dropout rate applied after adding the positional encodings.
    max_len : int, default=5000
        Maximum sequence length for which encodings are pre-computed.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        nnDropout = _safe_import("torch.nn.Dropout")
        self.dropout = nnDropout(p=dropout)

        torch_zeros = _safe_import("torch.zeros")
        torch_arange = _safe_import("torch.arange")
        torch_float = _safe_import("torch.float")
        torch_exp = _safe_import("torch.exp")
        torch_sin = _safe_import("torch.sin")
        torch_cos = _safe_import("torch.cos")

        pe = torch_zeros(max_len, d_model)
        position = torch_arange(0, max_len, dtype=torch_float).unsqueeze(1)
        div_term = torch_exp(
            torch_arange(0, d_model).float() * (-math.log(10000.0) / d_model)
        )
        pe += torch_sin(position * div_term)
        pe += torch_cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, seqlen=10, pos=0):
        """Add positional encodings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, n_feats)
            Input tensor containing the embedded time series.
        seqlen : int, default=10
            Sequence length used to determine along which axis to index.
        pos : int, default=0
            Offset into the pre-computed positional encodings.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``, with positional encodings added.
        """
        idx = 0 if x.shape[0] == seqlen else 1
        x = x + self.pe[pos : pos + x.size(idx), :]
        return self.dropout(x)


class SimplePatch(NNModule):
    """Non-deformable patch embedding layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels, i.e., variables of the time series.
    out_channels : int
        Number of output channels, i.e., the embedding dimension.
    seq_len : int
        Length of the input series.
    patch_size : int
        Size of the patches the series is split into.
    stride : int, default=1
        Stride between consecutive patches.
    norm : str, default="batch"
        Normalization applied to the patch embeddings, ``"batch"`` or ``"layer"``.
    padding_tp : str or None, default=None
        If ``"same"``, the output length equals the input length.
    device : str, default="cpu"
        Device the patch network is moved to.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        seq_len,
        patch_size,
        stride=1,
        norm="batch",
        padding_tp=None,
        device="cpu",
    ):
        super().__init__()

        nnSequential = _safe_import("torch.nn.Sequential")
        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnLayerNorm = _safe_import("torch.nn.LayerNorm")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")

        self.ptw = patch_size

        if padding_tp == "same":
            stride = 1
            self.l_pad = 1 * (patch_size - 1) // 2
            self.r_pad = 1 * (patch_size - 1) - self.l_pad
            n_padding = self.l_pad + self.r_pad

        else:
            n_stride = (seq_len - self.ptw) // stride + 1
            n_padding = n_stride * stride + self.ptw - seq_len
            self.l_pad = n_padding // 2
            self.r_pad = n_padding - self.l_pad

        self.new_len = seq_len + n_padding
        self.patch_net = (
            nnSequential(
                nnConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.ptw,
                    stride=stride,
                    padding=0,
                ),
            )
            .to(device)
            .apply(weights_init)
        )

        self.norm_tp = norm
        self.norm = (
            nnLayerNorm(out_channels)
            if norm == "layer"
            else nnBatchNorm1d(out_channels)
        )

    def forward(self, X):
        """Embed the input series into patches.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_len, n_channels)
            Input tensor containing the time series data.

        Returns
        -------
        torch.Tensor of shape (batch_size, out_channels, patch_count)
            The normalized patch embeddings.
        """
        pad = _torch_op("torch.nn.functional.pad")

        X = X.permute(0, 2, 1)
        X = pad(X, (self.l_pad, self.r_pad), mode="constant", value=0)
        X = self.patch_net(X)

        if self.norm_tp == "layer":
            X = self.norm(X.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            X = self.norm(X)
        return X


class BoxCoder(NNModule):
    """Decode predicted offsets into patch boundaries and sampling locations.

    Parameters
    ----------
    patch_count : int
        Number of patches the series is split into.
    patch_stride : int
        Stride between consecutive patches.
    patch_size : int
        Size of the patches the series is split into.
    seq_len : int
        Length of the padded input series.
    channels : int
        Number of input channels, i.e., variables of the time series.
    weights : tuple of float, default=(1.0, 1.0)
        Scaling weights, retained for compatibility with the reference
        implementation.
    tanh : bool, default=False
        Whether to squash the predicted offsets with ``tanh`` before decoding.
    device : str, default="cpu"
        Device the pre-computed patch anchors are placed on.
    """

    def __init__(
        self,
        patch_count,
        patch_stride,
        patch_size,
        seq_len,
        channels,
        weights=(1.0, 1.0),
        tanh=False,
        device="cpu",
    ):
        super().__init__()

        self.tanh = tanh
        self.seq_len = seq_len
        self.channels = channels
        self.patch_size = patch_size
        self.patch_count = patch_count
        self.patch_stride = patch_stride

        self._generate_anchor(device=device)
        self.weights = weights

    # compute the center points
    def _generate_anchor(self, device="cpu"):
        as_tensor = _safe_import("torch.as_tensor")

        anchors = []
        self.S_bias = (self.patch_size - 1) / 2

        for i in range(self.patch_count):
            x = i * self.patch_stride + 0.5 * (self.patch_size - 1)
            anchors.append(x)
        anchors = as_tensor(anchors, device=device)
        self.register_buffer("anchor", anchors)

    def forward(self, boxes):
        """Decode offsets into sampling locations and patch boundaries.

        Parameters
        ----------
        boxes : torch.Tensor of shape (batch_size, patch_count, 2)
            Predicted patch offsets.

        Returns
        -------
        points : torch.Tensor of shape
            (batch_size, patch_count, channels, patch_size, 2)
            Sampling locations in normalized ``[0, 1]`` coordinates.
        bound : torch.Tensor of shape (batch_size, patch_count, 2)
            Left and right boundary of every patch.
        """
        # bound is kept local, not stored on the module: caching an intermediate,
        # non-leaf tensor as an attribute breaks deepcopy of the fitted module
        bound = self.decode(boxes)  # (bs, patch_count, 2)
        points = self.meshgrid(bound)
        return points, bound

    def decode(self, rel_codes):
        """Turn relative offsets into left and right patch bounds in ``[0, 1]``.

        Parameters
        ----------
        rel_codes : torch.Tensor of shape (batch_size, patch_count, 2)
            Predicted patch offsets.

        Returns
        -------
        torch.Tensor of shape (batch_size, patch_count, 2)
            Left and right boundary of every patch.
        """
        torch_tanh = _torch_op("torch.tanh")
        torch_relu = _torch_op("torch.relu")
        zeros_like = _torch_op("torch.zeros_like")

        boxes = self.anchor

        if self.tanh:
            dx = torch_tanh(rel_codes[:, :, 0])
            ds = torch_relu(torch_tanh(rel_codes[:, :, 1]) + self.S_bias)

        else:
            dx = rel_codes[:, :, 0]
            ds = torch_relu(rel_codes[:, :, 1] + self.S_bias)

        pred_boxes = zeros_like(rel_codes)
        ref_x = boxes.view(1, boxes.shape[0])

        # dx, ds: (bs, patch_count, 1)
        # ref_x: (1, patch_count)
        pred_boxes[:, :, 0] = dx + ref_x - ds
        pred_boxes[:, :, 1] = dx + ref_x + ds
        pred_boxes /= self.seq_len - 1

        pred_boxes = pred_boxes.clamp_(min=0.0, max=1.0)
        return pred_boxes

    def meshgrid(self, boxes):
        """Expand patch bounds into a sampling grid.

        Parameters
        ----------
        boxes : torch.Tensor of shape (batch_size, patch_count, 2)
            Left and right boundary of every patch.

        Returns
        -------
        torch.Tensor of shape (batch_size, patch_count, channels, patch_size, 2)
            Sampling locations for ``torch.nn.functional.grid_sample``.
        """
        interpolate = _torch_op("torch.nn.functional.interpolate")
        zeros_like = _torch_op("torch.zeros_like")
        stack = _torch_op("torch.stack")

        B = boxes.shape[0]
        channel_boxes = zeros_like(boxes)
        channel_boxes[:, :, 1] = 1.0

        xs = interpolate(boxes, size=self.patch_size, mode="linear", align_corners=True)
        ys = interpolate(
            channel_boxes, size=self.channels, mode="linear", align_corners=True
        )
        # xs: [bs, patch_count, patch_size]  ys: [bs, patch_count, channels(also feats)]

        xs = xs.unsqueeze(2).expand(B, self.patch_count, self.channels, self.patch_size)
        ys = ys.unsqueeze(3).expand(B, self.patch_count, self.channels, self.patch_size)

        grid = stack([xs, ys], dim=-1)
        return grid  # [bs, patch_count, channel, patch_size, 2]


class OffsetPredictor(NNModule):
    """Predict the patch offsets used by :class:`DeformablePatch`.

    Parameters
    ----------
    in_feats : int
        Number of input channels, i.e., variables of the time series.
    patch_size : int
        Size of the patches the series is split into.
    stride : int
        Stride between consecutive patches.
    activation_hidden : torch.nn.Module or None, default=None
        Activation applied between the hidden layers of the offset predictor.
        If None, no activation is applied.
    mod : int, default=0
        Architecture of the offset predictor. ``0`` uses two convolutions with an
        activation in between, ``1`` a single convolution, and ``2`` an MLP.
    """

    def __init__(self, in_feats, patch_size, stride, activation_hidden=None, mod=0):
        super().__init__()

        nnSequential = _safe_import("torch.nn.Sequential")
        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnLinear = _safe_import("torch.nn.Linear")
        nnIdentity = _safe_import("torch.nn.Identity")

        self.mod = mod
        self.stride = stride
        self.in_feats = in_feats
        self.patch_size = patch_size

        act = nnIdentity() if activation_hidden is None else activation_hidden

        if mod == 0:
            self.offset_predictor = nnSequential(
                nnConv1d(in_feats, 64, patch_size, stride=stride, padding=0),
                act,
                nnConv1d(64, 2, 1, 1, padding=0),
            )
        elif mod == 1:  # Single Conv
            self.offset_predictor = nnSequential(
                nnConv1d(in_feats, 2, patch_size, stride=stride, padding=0),
            )
        elif mod == 2:  # MLP
            # channel independence
            in_1, in_2, in_3 = (
                patch_size * in_feats,
                2 * (patch_size * in_feats) // 3,
                (patch_size * in_feats) // 3,
            )
            out_1, out_2, out_3 = in_2, in_3, 2

            self.offset_predictor = nnSequential(
                nnLinear(in_1, out_1),
                act,
                nnLinear(in_2, out_2),
                act,
                nnLinear(in_3, out_3),
            )

    def forward(self, X):
        """Predict the patch offsets.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_channels, seq_len)
            Input tensor containing the padded time series data.

        Returns
        -------
        torch.Tensor of shape (batch_size, patch_count, 2)
            Predicted patch offsets.
        """
        if self.mod in [0, 1]:
            pred_offset = self.offset_predictor(X).permute(0, 2, 1)
        else:
            unfold = _torch_op("torch.nn.functional.unfold")
            patch_X = unfold(
                X.unsqueeze(1),
                kernel_size=(self.in_feats, self.patch_size),
                stride=(1, self.stride),
            )
            pred_offset = self.offset_predictor(patch_X.permute(0, 2, 1))

        return pred_offset  # (bs, patch_count, 2)


class DeformablePatch(NNModule):
    """Deformable patch embedding layer of ConvTimeNet.

    Parameters
    ----------
    in_feats : int
        Number of input channels, i.e., variables of the time series.
    out_feats : int
        Number of output channels, i.e., the embedding dimension.
    seq_len : int
        Length of the input series.
    patch_size : int
        Size of the patches the series is split into.
    stride : int
        Stride between consecutive patches.
    padding_tp : str or None, default=None
        If ``"same"``, the number of patches equals the input length.
    norm : str, default="batch"
        Normalization applied to the patch embeddings, ``"batch"`` or ``"layer"``.
    activation_hidden : torch.nn.Module or None, default=None
        Activation applied to the patch embeddings and inside the offset
        predictor. If None, no activation is applied.
    offset_mod : int, default=0
        Architecture of the offset predictor, see :class:`OffsetPredictor`.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        seq_len,
        patch_size,
        stride,
        padding_tp=None,
        norm="batch",
        activation_hidden=None,
        offset_mod=0,
    ):
        super().__init__()

        nnDropout = _safe_import("torch.nn.Dropout")
        nnConv2d = _safe_import("torch.nn.Conv2d")
        nnLayerNorm = _safe_import("torch.nn.LayerNorm")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")

        if padding_tp == "same":
            stride = 1
            self.patch_count = seq_len
            l_pad = 1 * (patch_size - 1) // 2
            r_pad = 1 * (patch_size - 1) - l_pad
            n_padding = l_pad + r_pad

        else:
            n_stride = (seq_len - patch_size) // stride + 1
            n_padding = n_stride * stride + patch_size - seq_len
            self.patch_count = n_stride + 1

        self.n_padding = n_padding

        self.patch_size = patch_size
        self.in_feats, self.out_feats = in_feats, out_feats

        self.dropout = nnDropout(0.1)
        self.new_len = seq_len + n_padding

        # offset predictor
        self.offset_predictor = OffsetPredictor(
            in_feats,
            patch_size,
            stride,
            activation_hidden=activation_hidden,
            mod=offset_mod,
        )

        self.box_coder = BoxCoder(
            self.patch_count, stride, patch_size, self.new_len, in_feats
        )

        # output layers
        self.output_conv = nnConv2d(1, self.out_feats, (self.in_feats, self.patch_size))
        self.norm_tp = norm
        self.output_act = activation_hidden
        self.norm = (
            nnLayerNorm(self.out_feats)
            if norm == "layer"
            else nnBatchNorm1d(self.out_feats)
        )

    def get_sampling_location(self, X):
        """Get sampling location.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_channels, seq_len)
            Input tensor containing the padded time series data.

        Returns
        -------
        sampling_locations : torch.Tensor of shape
            (batch_size, patch_count, n_channels, patch_size, 2)
            Sampling locations in normalized ``[0, 1]`` coordinates.
        bound : torch.Tensor of shape (batch_size, patch_count, 2)
            Left and right boundary of every patch.
        """
        # get offset
        pred_offset = self.offset_predictor(X)
        sampling_locations, bound = self.box_coder(pred_offset)
        return sampling_locations, bound

    def forward(self, X, return_bound=False):
        """Embed the input series into deformable patches.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_len, n_channels)
            Input tensor containing the time series data.
        return_bound : bool, default=False
            Whether to also return the padded input and the patch boundaries.

        Returns
        -------
        torch.Tensor of shape (batch_size, out_feats, patch_count)
            The normalized patch embeddings. If ``return_bound`` is True, a
            2-tuple of the embeddings and ``[padded input, patch boundaries]``.
        """
        pad = _torch_op("torch.nn.functional.pad")
        grid_sample = _torch_op("torch.nn.functional.grid_sample")

        X = X.permute(0, 2, 1)
        X = pad(X, (0, self.n_padding), mode="constant", value=0)

        # Consider the X as img.shape: (B, C, H, W) <--> (bs,1,channel,padded_window)
        img = X.unsqueeze(1)
        B = img.shape[0]

        # sampling_locations: [bs, patch_count, channel, patch_size, 2]
        sampling_locations, bound = self.get_sampling_location(X)
        sampling_locations = sampling_locations.view(
            B, self.patch_count * self.in_feats, self.patch_size, 2
        )

        sampling_locations = (sampling_locations - 0.5) * 2  # location map: [-1, 1]
        output = grid_sample(img, sampling_locations, align_corners=True)
        output = output.view(
            B, self.patch_count, self.in_feats, self.patch_size
        )  # (B, patch_count, channel, patch_size)

        # output_proj
        output = output.permute(0, 1, 3, 2).contiguous()
        output = output.view(B * self.patch_count, 1, self.in_feats, self.patch_size)
        output = self.output_conv(output)  # (bs*patch_count, out_feats, 1, 1)
        output = output.view(B, self.patch_count, self.out_feats)

        output = self.output_act(output) if self.output_act is not None else output
        if self.norm_tp == "layer":
            output = self.norm(output).permute(0, 2, 1)
        else:
            output = self.norm(output.permute(0, 2, 1))

        if return_bound:
            return output, [X, bound]
        else:
            return output
