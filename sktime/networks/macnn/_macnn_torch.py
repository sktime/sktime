"""Multi-scale Attention Convolutional Neural Network (MACNN) in PyTorch."""

__authors__ = ["Faakhir30"]
__all__ = ["MACNNNetworkTorch"]


from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class MACNNNetworkTorch(NNModule):
    """Multi-Scale Attention Convolutional Neural Network (MACNN) in PyTorch.

    Implements a multi-scale attention mechanism that learns feature representations
    across different temporal scales.

    Parameters
    ----------
    input_size : int or tuple
        Number of expected features in the input.
    num_classes : int
        Number of classes to predict
    padding : str, default="same"
        The type of padding to be provided in MACNN Blocks.
        Used for pooling layers only. Convolution layers always use "same" padding,
        so that multi-scale outputs can be concatenated.
    pool_size : int, default=3
        A single value representing pooling windows which are applied
        between two MACNN Blocks.
    strides : int, default=2
        A single value representing strides to be taken during the
        pooling operation.
    repeats : int, default=2
        The number of MACNN Blocks to be stacked.
    filter_sizes : tuple of int, default=(64, 128, 256)
        The filter sizes of Conv1D layers within each MACNN Block.
    kernel_size : tuple of int, default=(3, 6, 12)
        The kernel sizes of Conv1D layers within each MACNN Block.
    reduction : int, default=16
        The factor by which the first dense layer of a MACNN Block will be divided by.
    activation : str or None = None
        Activation function used for output layer.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu'
    activation_hidden : str, default="relu"
        Activation function used for internal layers.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu'
    random_state : int, default=0
        Seed to ensure reproducibility

    Notes
    -----
    Based on the paper by Wei Chen et al., 2021

    References
    ----------
    .. [1] Wei Chen et. al, Multi-scale Attention Convolutional
    Neural Network for time series classification,
    Neural Networks, Volume 136, 2021, Pages 126-140, ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2021.01.001.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["jnrusson1", "noxthot"],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_size: int | tuple,
        num_classes: int,
        padding: str = "same",
        pool_size: int = 3,
        strides: int = 2,
        repeats: int = 2,
        filter_sizes: tuple = (64, 128, 256),
        kernel_size: tuple = (3, 6, 12),
        reduction: int = 16,
        activation: str | None = None,
        activation_hidden: str = "relu",
        random_state: int = 0,
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides
        self.repeats = repeats
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.random_state = random_state

        super().__init__()

        if self.activation is not None:
            self._activation = _instantiate_activation(self.activation)
        self._activation_hidden = _instantiate_activation(self.activation_hidden)

        nnMaxPool1d = _safe_import("torch.nn.MaxPool1d")
        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")

        self.macnn_stack1 = self._build_macnn_stack(
            self.input_size, self.filter_sizes[0], repeats
        )
        self.pool1 = nnMaxPool1d(
            kernel_size=self.pool_size, stride=self.strides, padding=self._get_padding()
        )

        self.macnn_stack2 = self._build_macnn_stack(
            self.filter_sizes[0] * len(self.kernel_size), self.filter_sizes[1], repeats
        )
        self.pool2 = nnMaxPool1d(
            kernel_size=self.pool_size, stride=self.strides, padding=self._get_padding()
        )

        self.macnn_stack3 = self._build_macnn_stack(
            self.filter_sizes[1] * len(self.kernel_size), self.filter_sizes[2], repeats
        )

        self.global_avg_pool = nnAdaptiveAvgPool1d(1)

        nnLinear = _safe_import("torch.nn.Linear")
        self.fc = nnLinear(self.filter_sizes[2] * len(self.kernel_size), num_classes)

        self.apply(self._init_weights)

    def _get_padding(self):
        """Convert padding string to integer for PyTorch."""
        if self.padding.lower() == "same":
            return self.pool_size // 2
        return 0

    def _build_macnn_stack(self, in_channels, filter_size, repeats):
        """Build a stack of MACNN blocks.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        filter_size : int
            Number of filters for the MACNN blocks
        repeats : int
            Number of MACNN blocks to stack

        Returns
        -------
        stack : torch.nn.Sequential
            Sequential container of MACNN blocks
        """
        nnSequential = _safe_import("torch.nn.Sequential")
        layers = []
        for i in range(repeats):
            in_ch = in_channels if i == 0 else filter_size * len(self.kernel_size)
            layers.append(
                MACNNBlock(
                    in_channels=in_ch,
                    filter_size=filter_size,
                    kernel_size=self.kernel_size,
                    reduction=self.reduction,
                    padding=self.padding,
                    activation=self.activation_hidden,
                )
            )
        return nnSequential(*layers)

    def forward(self, X):
        """Forward pass through the MACNN network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_length, n_dims)
            Input tensor containing the time series data.
            Note: Data comes from PytorchDataset already transposed to
            (batch, seq_length, n_dims) format.

        Returns
        -------
        out : torch.Tensor of shape (batch_size, num_classes)
            Output tensor containing the class predictions.
        """
        X = X.transpose(1, 2)  # (batch, channels, length)

        # First stack
        x = self.macnn_stack1(X)
        x = self.pool1(x)

        # Second stack
        x = self.macnn_stack2(x)
        x = self.pool2(x)

        # Third stack
        x = self.macnn_stack3(x)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)

        out = self.fc(x)

        # Squeeze output (if regression)
        if self.num_classes == 1:
            out = out.squeeze(-1)

        return out

    def _init_weights(self, module):
        """Apply initialization to weights.

        For Conv layers: He uniform initialization

        Parameters
        ----------
        module : torch.nn.Module
            Input module on which to apply initializations.
        """
        nnInitKaiming_uniform_ = _safe_import("torch.nn.init.kaiming_uniform_")
        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnLinear = _safe_import("torch.nn.Linear")

        if isinstance(module, nnConv1d):
            nnInitKaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.fill_(0)
        elif isinstance(module, nnLinear):
            nnInitKaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.fill_(0)


class MACNNBlock(NNModule):
    """A single MACNN Block with multi-scale attention mechanism."""

    def __init__(
        self,
        in_channels: int,
        filter_size: int,
        kernel_size: tuple,
        reduction: int,
        padding: str,
        activation: str,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.padding = padding
        self.activation_str = activation

        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnLinear = _safe_import("torch.nn.Linear")
        nnModuleList = _safe_import("torch.nn.ModuleList")

        # Multi-scale convolutions
        # NOTE: The `padding` parameter controls the pooling layer behavior,
        # not convolution. For multi-scale convolutions to concatenate,
        # all kernels MUST produce the same output length.
        # Therefore, we always use padding="same" for Conv1d layers.
        self.convs = nnModuleList()
        for ks in kernel_size:
            self.convs.append(nnConv1d(in_channels, filter_size, ks, padding="same"))

        out_channels = filter_size * len(kernel_size)
        self.bn = nnBatchNorm1d(out_channels)

        self._activation = _instantiate_activation(self.activation_str)

        self.global_pool = _safe_import("torch.nn.AdaptiveAvgPool1d")(1)

        reduced_dim = max(1, out_channels // reduction)
        self.fc1 = nnLinear(out_channels, reduced_dim)
        self.fc2 = nnLinear(reduced_dim, out_channels)

    def forward(self, x):
        """Forward pass through MACNN block."""
        torchCat = _safe_import("torch.cat")

        # Multi-scale convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))

        x1 = torchCat(conv_outputs, dim=1)

        x1 = self.bn(x1)
        x1 = self._activation(x1)

        x2 = self.global_pool(x1)
        x2 = x2.squeeze(-1)  # (B, C)

        x2 = self.fc1(x2)
        x2 = self._activation(x2)
        x2 = self.fc2(x2)
        x2 = self._activation(x2)

        x2 = x2.unsqueeze(-1)  # (B, C, 1)
        out = x1 * x2

        return out


def _instantiate_activation(activation):
    """Instantiate the activation function for hidden layers.

    Returns
    -------
    activation_function : torch.nn.Module
        The activation function to be applied in hidden layers.
    """
    if isinstance(activation, NNModule):
        return activation
    elif isinstance(activation, str):
        act = activation.lower()
        if act == "relu":
            return _safe_import("torch.nn.ReLU")()
        elif act == "tanh":
            return _safe_import("torch.nn.Tanh")()
        elif act == "sigmoid":
            return _safe_import("torch.nn.Sigmoid")()
        elif act == "leaky_relu":
            return _safe_import("torch.nn.LeakyReLU")()
        elif act == "elu":
            return _safe_import("torch.nn.ELU")()
        elif act == "selu":
            return _safe_import("torch.nn.SELU")()
        elif act == "gelu":
            return _safe_import("torch.nn.GELU")()
        else:
            raise ValueError(
                "If `activation` is a string, it must be one of "
                "'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', or 'gelu'. "
                f"Found {activation}"
            )
    else:
        raise TypeError(
            "`activation` should either be of type str or torch.nn.Module. "
            f"But found the type to be: {type(activation)}"
        )
