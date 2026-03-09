"""Time Series Attentional Prototype Network (TapNet) in PyTorch.

PyTorch implementation of the TapNet network architecture,
migrated from the original TensorFlow/Keras implementation.
"""

__authors__ = ["dakshhhhh16"]
__all__ = ["TapNetNetworkTorch"]

import math

import numpy as np

from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class _SeqSelfAttentionTorch(NNModule):
    """Sequential self-attention layer in PyTorch.

    Supports both additive and multiplicative attention types.
    This is a PyTorch re-implementation of the Keras SeqSelfAttention layer
    used in the original TapNet.

    Parameters
    feature_dim : int
        The dimension of input features (last axis of input tensor).
    units : int, default=32
        The dimension of vectors used to calculate attention weights.
    attention_type : str, default='multiplicative'
        'additive' or 'multiplicative'.
    use_attention_bias : bool, default=True
        Whether to use bias in attention calculations.
    """

    ATTENTION_TYPE_ADD = "additive"
    ATTENTION_TYPE_MUL = "multiplicative"

    def __init__(
        self,
        feature_dim,
        units=32,
        attention_type="multiplicative",
        use_attention_bias=True,
    ):
        super().__init__()
        self.units = units
        self.attention_type = attention_type
        self.use_attention_bias = use_attention_bias

        nnLinear = _safe_import("torch.nn.Linear")
        nnParameter = _safe_import("torch.nn.Parameter")
        torchZeros = _safe_import("torch.zeros")

        if attention_type == self.ATTENTION_TYPE_ADD:
            self.Wt = nnLinear(feature_dim, units, bias=False)
            self.Wx = nnLinear(feature_dim, units, bias=False)
            self.Wa = nnLinear(units, 1, bias=False)
            if use_attention_bias:
                self.ba = nnParameter(torchZeros(1))
        elif attention_type == self.ATTENTION_TYPE_MUL:
            self.Wa = nnLinear(feature_dim, feature_dim, bias=False)
            if use_attention_bias:
                self.ba = nnParameter(torchZeros(1))
        else:
            raise NotImplementedError(
                f"No implementation for attention type: {attention_type}"
            )

    def forward(self, inputs):
        """Forward pass through self-attention.

        Parameters
        inputs : torch.Tensor of shape (batch_size, seq_length, feature_dim)
            Input tensor.

        Returns
        v : torch.Tensor of shape (batch_size, seq_length, feature_dim)
            Attention-weighted output.
        """
        torchSoftmax = _safe_import("torch.nn.functional.softmax")
        torchBmm = _safe_import("torch.bmm")
        torchTanh = _safe_import("torch.tanh")

        if self.attention_type == self.ATTENTION_TYPE_ADD:
            e = self._additive_attention(inputs, torchTanh)
        else:
            e = self._multiplicative_attention(inputs, torchBmm)

        # a = softmax(e)
        a = torchSoftmax(e, dim=-1)
        # v = a @ inputs
        v = torchBmm(a, inputs)
        return v

    def _additive_attention(self, inputs, torchTanh):
        """Compute additive attention scores.

        Parameters
        inputs : torch.Tensor of shape (batch_size, seq_length, feature_dim)
        torchTanh : callable

        Returns
        e : torch.Tensor of shape (batch_size, seq_length, seq_length)
        """
        # q: (batch, seq, units), k: (batch, seq, units)
        q = self.Wt(inputs).unsqueeze(2)  # (batch, seq, 1, units)
        k = self.Wx(inputs).unsqueeze(1)  # (batch, 1, seq, units)
        h = torchTanh(q + k)  # (batch, seq, seq, units)
        e = self.Wa(h).squeeze(-1)  # (batch, seq, seq)
        if self.use_attention_bias:
            e = e + self.ba
        return e

    def _multiplicative_attention(self, inputs, torchBmm):
        """Compute multiplicative attention scores.

        Parameters
        inputs : torch.Tensor of shape (batch_size, seq_length, feature_dim)
        torchBmm : callable

        Returns
        e : torch.Tensor of shape (batch_size, seq_length, seq_length)
        """
        # e = inputs @ Wa @ inputs^T
        q = self.Wa(inputs)  # (batch, seq, feature_dim)
        e = torchBmm(q, inputs.transpose(1, 2))  # (batch, seq, seq)
        if self.use_attention_bias:
            e = e + self.ba
        return e


class TapNetNetworkTorch(NNModule):
    """Time series Attentional Prototype Network (TapNet) in PyTorch.

    Adapted from the TensorFlow/Keras implementation in [1]_.

    Parameters
    n_channels : int
        Number of input channels (dimensions) of the time series.
    series_length : int
        Length of the input time series.
    activation : str, default = "leaky_relu"
        Activation function used in the hidden layers.
    kernel_size : tuple of int, default = (8, 5, 3)
        Specifying the length of the 1D convolution window.
    layers : tuple of int, default = (500, 300)
        Size of dense layers in the mapping section.
    filter_sizes : tuple of int, default = (256, 256, 128)
        Number of filters in each convolutional block.
    random_state : int or None, default = None
        Seed for random number generation.
    rp_params : tuple of int, default = (-1, 3)
        Parameters for random permutation: (rp_group, rp_dim).
        If rp_group < 0, it is set to 3 and rp_dim to floor(n_channels * 2/3).
    dropout : float, default = 0.5
        Dropout rate for the CNN branch.
    lstm_dropout : float, default = 0.8
        Dropout rate for the LSTM branch.
    dilation : int, default = 1
        Dilation value for convolutional layers.
    padding : str, default = "same"
        Type of padding for convolution layers.
    use_rp : bool, default = True
        Whether to use random projections (channel subsampling).
    use_att : bool, default = True
        Whether to use self-attention.
    use_lstm : bool, default = True
        Whether to use an LSTM layer.
    use_cnn : bool, default = True
        Whether to use a CNN branch.
    num_classes : int, default = 1
        Number of output classes (1 for regression).
    activation_output : str or None, default = None
        Activation function for the output layer.
        If None, no activation is applied.
    use_bias : bool, default = True
        Whether to use bias in the output dense layer.

    References
    .. [1] Zhang et al. Tapnet: Multivariate time series classification with
    attentional prototypical network,
    Proceedings of the AAAI Conference on Artificial Intelligence
    34(4), 6845-6852, 2020
    """

    _tags = {
        "authors": ["dakshpathak"],
        "maintainers": ["dakshpathak"],
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        n_channels,
        series_length,
        activation="leaky_relu",
        kernel_size=(8, 5, 3),
        layers=(500, 300),
        filter_sizes=(256, 256, 128),
        random_state=None,
        rp_params=(-1, 3),
        dropout=0.5,
        lstm_dropout=0.8,
        dilation=1,
        padding="same",
        use_rp=True,
        use_att=True,
        use_lstm=True,
        use_cnn=True,
        num_classes=1,
        activation_output=None,
        use_bias=True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.series_length = series_length
        self.activation_name = activation
        self.kernel_size = kernel_size
        self.layers_sizes = layers
        self.filter_sizes = filter_sizes
        self.random_state = random_state
        self.rp_params = rp_params
        self.dropout_rate = dropout
        self.lstm_dropout_rate = lstm_dropout
        self.dilation = dilation
        self.padding = padding
        self.use_rp = use_rp
        self.use_att = use_att
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.num_classes = num_classes
        self.activation_output = activation_output
        self.use_bias = use_bias

        # Set random state
        if self.random_state is not None:
            torchManual_seed = _safe_import("torch.manual_seed")
            torchManual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Resolve random projection parameters
        if self.rp_params[0] < 0:
            self.rp_group = 3
            self.rp_dim = max(1, math.floor(self.n_channels * 2 / 3))
        else:
            self.rp_group, self.rp_dim = self.rp_params

        # Import torch modules
        nnLSTM = _safe_import("torch.nn.LSTM")
        nnDropout = _safe_import("torch.nn.Dropout")
        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnLinear = _safe_import("torch.nn.Linear")
        nnModuleList = _safe_import("torch.nn.ModuleList")
        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")

        self._activation = self._get_activation(self.activation_name)

        # ---- LSTM branch ----
        if self.use_lstm:
            self.lstm_dim = 128
            # Input: (batch, seq_len, n_channels)
            self.lstm = nnLSTM(
                input_size=self.n_channels,
                hidden_size=self.lstm_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            self.lstm_dropout = nnDropout(self.lstm_dropout_rate)
            if self.use_att:
                self.lstm_attention = _SeqSelfAttentionTorch(
                    feature_dim=self.lstm_dim,
                    units=128,
                    attention_type="multiplicative",
                )
            self.lstm_gap = nnAdaptiveAvgPool1d(1)

        #  CNN branch 
        if self.use_cnn:
            if self.use_rp:
                # Generate random projection indices for each group
                self.rp_indices = []
                for _ in range(self.rp_group):
                    idx = np.random.permutation(self.n_channels)[: self.rp_dim]
                    self.rp_indices.append(idx)

                # Create conv blocks for each random projection group
                self.cnn_blocks = nnModuleList()
                self.cnn_bns = nnModuleList()
                self.cnn_attentions = nnModuleList()
                self.cnn_gaps = nnModuleList()

                for _ in range(self.rp_group):
                    block_convs = nnModuleList()
                    block_bns = nnModuleList()

                    # 3 conv layers per block
                    in_ch = self.rp_dim
                    for j in range(3):
                        pad = self._compute_padding(self.kernel_size[j])
                        conv = nnConv1d(
                            in_channels=in_ch,
                            out_channels=self.filter_sizes[j],
                            kernel_size=self.kernel_size[j],
                            stride=1,
                            dilation=self.dilation,
                            padding=pad,
                        )
                        bn = nnBatchNorm1d(self.filter_sizes[j])
                        block_convs.append(conv)
                        block_bns.append(bn)
                        in_ch = self.filter_sizes[j]

                    self.cnn_blocks.append(block_convs)
                    self.cnn_bns.append(block_bns)

                    if self.use_att:
                        att = _SeqSelfAttentionTorch(
                            feature_dim=self.filter_sizes[2],
                            units=128,
                            attention_type="multiplicative",
                        )
                        self.cnn_attentions.append(att)
                    else:
                        self.cnn_attentions.append(None)

                    self.cnn_gaps.append(nnAdaptiveAvgPool1d(1))
            else:
                # Single CNN path (no random projections)
                self.cnn_convs = nnModuleList()
                self.cnn_bns_single = nnModuleList()

                in_ch = self.n_channels
                for j in range(3):
                    pad = self._compute_padding(self.kernel_size[j])
                    conv = nnConv1d(
                        in_channels=in_ch,
                        out_channels=self.filter_sizes[j],
                        kernel_size=self.kernel_size[j],
                        stride=1,
                        dilation=self.dilation,
                        padding=pad,
                    )
                    bn = nnBatchNorm1d(self.filter_sizes[j])
                    self.cnn_convs.append(conv)
                    self.cnn_bns_single.append(bn)
                    in_ch = self.filter_sizes[j]

                if self.use_att:
                    self.cnn_attention_single = _SeqSelfAttentionTorch(
                        feature_dim=self.filter_sizes[2],
                        units=128,
                        attention_type="multiplicative",
                    )
                self.cnn_gap_single = nnAdaptiveAvgPool1d(1)

        #  Mapping / Dense layers 
        # Compute combined feature dimension
        combined_dim = 0
        if self.use_cnn:
            if self.use_rp:
                combined_dim += self.filter_sizes[2] * self.rp_group
            else:
                combined_dim += self.filter_sizes[2]
        if self.use_lstm:
            combined_dim += self.lstm_dim

        self.fc1 = nnLinear(combined_dim, self.layers_sizes[0])
        self.bn_fc = nnBatchNorm1d(self.layers_sizes[0])
        self.fc2 = nnLinear(self.layers_sizes[0], self.layers_sizes[1])

        # Output layer 
        self.output_layer = nnLinear(
            self.layers_sizes[1], self.num_classes, bias=self.use_bias
        )
        if self.activation_output is not None:
            self._output_activation = self._get_output_activation(
                self.activation_output
            )
        else:
            self._output_activation = None

    def _compute_padding(self, kernel_size):
        """Compute padding size for 'same' padding with dilation.

        Parameters
        kernel_size : int
            Size of the convolutional kernel.

        Returns
        pad : int or str
            Padding value.
        """
        if self.padding == "same":
            # For dilation, effective kernel size = dilation * (kernel_size - 1) + 1
            # 'same' padding = (effective_kernel_size - 1) // 2
            effective_ks = self.dilation * (kernel_size - 1) + 1
            return (effective_ks - 1) // 2
        elif self.padding == "valid":
            return 0
        else:
            return 0

    def _get_activation(self, name):
        """Get activation function by name.

        Parameters
        name : str
            Name of activation function.

        Returns
        activation : callable
            Activation function.
        """
        if name.lower() == "relu":
            return _safe_import("torch.nn.ReLU")()
        elif name.lower() in ("leaky_relu", "leakyrelu"):
            return _safe_import("torch.nn.LeakyReLU")()
        elif name.lower() == "sigmoid":
            return _safe_import("torch.nn.Sigmoid")()
        elif name.lower() == "tanh":
            return _safe_import("torch.nn.Tanh")()
        else:
            return _safe_import("torch.nn.LeakyReLU")()

    def _get_output_activation(self, name):
        """Get output activation function by name.

        Parameters
        name : str
            Name of activation function.

        Returns
        activation : callable
            Activation function module.
        """
        if isinstance(name, NNModule):
            return name
        elif isinstance(name, str):
            name_lower = name.lower()
            if name_lower == "sigmoid":
                return _safe_import("torch.nn.Sigmoid")()
            elif name_lower == "softmax":
                return _safe_import("torch.nn.Softmax")(dim=-1)
            elif name_lower == "relu":
                return _safe_import("torch.nn.ReLU")()
            elif name_lower == "logsoftmax":
                return _safe_import("torch.nn.LogSoftmax")(dim=-1)
            elif name_lower == "logsigmoid":
                return _safe_import("torch.nn.LogSigmoid")()
            elif name_lower == "linear":
                return None  # identity
            else:
                raise ValueError(
                    f"Unknown output activation: {name}. "
                    "Supported: 'sigmoid', 'softmax', 'relu', 'logsoftmax', "
                    "'logsigmoid', 'linear', or None."
                )
        else:
            raise TypeError(
                f"`activation_output` should be str, None, or nn.Module. "
                f"Got {type(name)}"
            )

    def forward(self, X):
        """Forward pass through TapNet.

        Parameters
        X : torch.Tensor of shape (batch_size, series_length, n_channels)
            Input time series data.
            Note: The base class PytorchDataset transposes from
            (batch, n_dims, series_length) to (batch, series_length, n_dims).

        Returns
        out : torch.Tensor
            Output tensor of shape (batch_size, num_classes).
        """
        torchCat = _safe_import("torch.cat")
        torchTensor = _safe_import("torch.tensor")
        torchLong = _safe_import("torch.long")

        features = []

        #LSTM branch  
        if self.use_lstm:
            x_lstm = self._forward_lstm(X)
            features.append(x_lstm)

        # CNN branch 
        if self.use_cnn:
            x_cnn = self._forward_cnn(X, torchCat, torchTensor, torchLong)
            features.append(x_cnn)

        # Combine features
        if len(features) > 1:
            x = torchCat(features, dim=-1)
        else:
            x = features[0]

        #  Mapping section 
        x = self.fc1(x)
        x = self._activation(x)
        x = self.bn_fc(x)
        x = self.fc2(x)

        #  Output layer 
        x = self.output_layer(x)
        if self._output_activation is not None:
            x = self._output_activation(x)

        # For regression (num_classes=1), squeeze the last dim
        if self.num_classes == 1:
            x = x.squeeze(-1)

        return x

    def _forward_lstm(self, X):
        """Process LSTM branch.

        Parameters
        X : torch.Tensor of shape (batch_size, series_length, n_channels)

        Returns
        x_lstm : torch.Tensor of shape (batch_size, lstm_dim)
        """
        # LSTM input: (batch, seq_len, input_size)
        x_lstm, _ = self.lstm(X)
        # x_lstm: (batch, seq_len, lstm_dim)
        x_lstm = self.lstm_dropout(x_lstm)

        if self.use_att:
            x_lstm = self.lstm_attention(x_lstm)
            # x_lstm: (batch, seq_len, lstm_dim)

        # Global Average Pooling: transpose to (batch, lstm_dim, seq_len)
        # then pool to (batch, lstm_dim, 1) then squeeze
        x_lstm = x_lstm.transpose(1, 2)
        x_lstm = self.lstm_gap(x_lstm).squeeze(-1)
        return x_lstm

    def _forward_cnn(self, X, torchCat, torchTensor, torchLong):
        """Process CNN branch.

        Parameters
        X : torch.Tensor of shape (batch_size, series_length, n_channels)
        torchCat : callable
        torchTensor : callable
        torchLong : torch dtype

        Returns
        x_cnn : torch.Tensor of shape (batch_size, feature_dim)
        """
        if self.use_rp:
            rp_outputs = []
            for i in range(self.rp_group):
                # Select random projection channels
                idx = self.rp_indices[i]
                # X: (batch, seq_len, n_channels) -> select channels
                channel = X[:, :, idx]  # (batch, seq_len, rp_dim)

                # Conv1d expects (batch, channels, seq_len)
                x_conv = channel.transpose(1, 2)

                # Apply 3 conv layers
                for j in range(3):
                    x_conv = self.cnn_blocks[i][j](x_conv)
                    x_conv = self.cnn_bns[i][j](x_conv)
                    x_conv = self._activation(x_conv)

                if self.use_att and self.cnn_attentions[i] is not None:
                    # Attention expects (batch, seq_len, features)
                    x_conv = x_conv.transpose(1, 2)
                    x_conv = self.cnn_attentions[i](x_conv)
                    x_conv = x_conv.transpose(1, 2)

                # Global Average Pooling
                x_conv = self.cnn_gaps[i](x_conv).squeeze(-1)
                rp_outputs.append(x_conv)

            x_cnn = torchCat(rp_outputs, dim=-1)
        else:
            # Single CNN path
            # Conv1d expects (batch, channels, seq_len)
            x_conv = X.transpose(1, 2)  # (batch, n_channels, seq_len)

            for j in range(3):
                x_conv = self.cnn_convs[j](x_conv)
                x_conv = self.cnn_bns_single[j](x_conv)
                x_conv = self._activation(x_conv)

            if self.use_att:
                # Attention: (batch, seq_len, features)
                x_conv = x_conv.transpose(1, 2)
                x_conv = self.cnn_attention_single(x_conv)
                x_conv = x_conv.transpose(1, 2)

            # Global Average Pooling
            x_cnn = self.cnn_gap_single(x_conv).squeeze(-1)

        return x_cnn
