"""InceptionTime in PyTorch."""

__authors__ = ["Faakhir30"]
__all__ = ["InceptionTimeNetworkTorch"]


from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class InceptionTimeNetworkTorch(NNModule):
    """InceptionTime network in PyTorch.

    Adapted from the implementation from Fawaz et al.
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

    Parameters
    ----------
    input_size : int
        Number of expected features in the input.
    num_classes : int
        Number of classes to predict
    n_filters : int, default=32
        Number of filters in the convolution layers
    use_residual : bool, default=True
        If True, uses residual connections
    use_bottleneck : bool, default=True
        If True, uses bottleneck layer in inception modules
    bottleneck_size : int, default=32
        Size of the bottleneck layer
    depth : int, default=6
        Number of inception modules to stack
    kernel_size : int, default=40
        Base kernel size for inception modules
    random_state : int or None, default=None
        Seed to ensure reproducibility
    activation : str or None, default=None
        Activation used for the final output layer.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu', None
    activation_hidden : str, default="relu"
        Activation function used for hidden layers (output from inception modules).
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu'
    activation_inception : str or None, default=None
        Activation function used inside the inception modules.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu', None

    Notes
    -----
    Network originally defined in:
    Fawaz et al., InceptionTime: Finding AlexNet for Time Series Classification,
    Data Mining and Knowledge Discovery, 34, 2020

    References
    ----------
    .. [1] Ismail Fawaz, Hassan and Lucas, Benjamin and Forestier, Germain and
    Pelletier, Charlotte and Schmidt, Daniel F. and Weber, Jonathan and Webb,
    Geoffrey I. and Idoumghar, Lhassane and Muller, Pierre-Alain and Petitjean,
    François, InceptionTime: Finding AlexNet for Time Series Classification,
    Data Mining and Knowledge Discovery, 34, 2020
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["hfawaz", "JamesLarge", "Withington", "noxthot"],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        n_filters: int = 32,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        bottleneck_size: int = 32,
        depth: int = 6,
        kernel_size: int = 40,
        random_state: int | None = None,
        activation: str | None = None,
        activation_hidden: str = "relu",
        activation_inception: str | None = None,
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.n_filters = n_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.random_state = random_state
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.activation_inception = activation_inception

        super().__init__()

        self._activation = self._instantiate_activation(self.activation)
        self._activation_hidden = self._instantiate_activation(self.activation_hidden)
        self._activation_inception = self._instantiate_activation(
            self.activation_inception
        )

        nnModuleList = _safe_import("torch.nn.ModuleList")
        self.inception_modules = nnModuleList()
        self.shortcut_layers = nnModuleList()

        current_channels = self.input_size
        for d in range(self.depth):
            self.inception_modules.append(
                InceptionModule(
                    in_channels=current_channels,
                    n_filters=self.n_filters,
                    kernel_size=self.kernel_size,
                    bottleneck_size=self.bottleneck_size,
                    use_bottleneck=self.use_bottleneck,
                    activation_inception=self._activation_inception,
                    activation_output=self._activation_hidden,
                )
            )
            current_channels = 4 * self.n_filters

            if self.use_residual and d % 3 == 2:
                if d == 2:  # First shortcut
                    shortcut_in_channels = self.input_size
                else:
                    shortcut_in_channels = 4 * self.n_filters

                self.shortcut_layers.append(
                    ShortcutLayer(
                        in_channels=shortcut_in_channels,
                        out_channels=current_channels,
                        activation=self._activation,
                    )
                )

        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")
        self.global_avg_pool = nnAdaptiveAvgPool1d(1)

        nnLinear = _safe_import("torch.nn.Linear")
        self.fc = nnLinear(current_channels, num_classes)

        self.apply(self._init_weights)

    def forward(self, X):
        """Forward pass through the InceptionTime network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_dims, seq_length)
            Input tensor containing the time series data.

        Returns
        -------
        out : torch.Tensor of shape (batch_size, num_classes)
            Output tensor containing the class predictions.
        """
        x = X
        input_res = X
        shortcut_idx = 0

        for d in range(self.depth):
            x = self.inception_modules[d](x)

            if self.use_residual and d % 3 == 2:
                x = self.shortcut_layers[shortcut_idx](input_res, x)
                input_res = x
                shortcut_idx += 1

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)

        out = self.fc(x)
        out = self._activation(out)

        if self.num_classes == 1:  # Regression case
            out = out.squeeze(-1)
        return out

    def _init_weights(self, module):
        """Apply initialization to weights.

        For Conv layers: He uniform initialization
        For Linear layers: He uniform initialization

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

    def _instantiate_activation(self, activation_str):
        """Instantiate activation function.

        Parameters
        ----------
        activation_str : str
            Name of the activation function

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied
        """
        if activation_str is None or activation_str.lower() == "linear":
            nnIdentity = _safe_import("torch.nn.Identity")
            return nnIdentity()
        elif activation_str.lower() == "relu":
            return _safe_import("torch.nn.ReLU")()
        elif activation_str.lower() == "tanh":
            return _safe_import("torch.nn.Tanh")()
        elif activation_str.lower() == "sigmoid":
            return _safe_import("torch.nn.Sigmoid")()
        elif activation_str.lower() == "leaky_relu":
            return _safe_import("torch.nn.LeakyReLU")()
        elif activation_str.lower() == "elu":
            return _safe_import("torch.nn.ELU")()
        elif activation_str.lower() == "selu":
            return _safe_import("torch.nn.SELU")()
        elif activation_str.lower() == "gelu":
            return _safe_import("torch.nn.GELU")()
        else:
            raise ValueError(
                f"Unsupported activation: {activation_str}. "
                "Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', "
                "'elu', 'selu', 'gelu', 'linear'"
            )


class InceptionModule(NNModule):
    """Single Inception Module.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    n_filters : int
        Number of filters for each convolution branch
    kernel_size : int
        Base kernel size (will be divided by powers of 2)
    bottleneck_size : int
        Size of the bottleneck layer
    use_bottleneck : bool
        If True, uses bottleneck layer
    activation_inception : torch.nn.Module
        Activation function used inside the inception module
    activation_output : torch.nn.Module
        Activation function used at the output of the module
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_size: int,
        bottleneck_size: int,
        use_bottleneck: bool,
        activation_inception,
        activation_output,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.use_bottleneck = use_bottleneck

        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnMaxPool1d = _safe_import("torch.nn.MaxPool1d")

        if use_bottleneck and in_channels > 1:
            self.bottleneck = nnConv1d(
                in_channels, bottleneck_size, kernel_size=1, padding=0, bias=False
            )
            self.bottleneck_activation = activation_inception
            conv_in_channels = bottleneck_size
        else:
            self.bottleneck = None
            conv_in_channels = in_channels

        kernel_sizes = [kernel_size // (2**i) for i in range(3)]

        self.conv_list = _safe_import("torch.nn.ModuleList")(
            [
                nnConv1d(
                    conv_in_channels,
                    n_filters,
                    kernel_size=ks,
                    padding="same",
                    bias=False,
                )
                for ks in kernel_sizes
            ]
        )

        self.conv_activations = _safe_import("torch.nn.ModuleList")(
            [activation_inception for _ in range(len(kernel_sizes))]
        )

        self.maxpool = nnMaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_from_maxpool = nnConv1d(
            in_channels, n_filters, kernel_size=1, padding=0, bias=False
        )
        self.maxpool_activation = activation_inception

        self.bn = nnBatchNorm1d(4 * n_filters)
        self.output_activation = activation_output

    def forward(self, x):
        """Forward pass through inception module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        out : torch.Tensor
            Output after applying inception module
        """
        if self.bottleneck is not None:
            input_inception = self.bottleneck(x)
            input_inception = self.bottleneck_activation(input_inception)
        else:
            input_inception = x

        # Convolutional branches
        conv_outputs = []
        for i, conv in enumerate(self.conv_list):
            out = conv(input_inception)
            out = self.conv_activations[i](out)
            conv_outputs.append(out)

        # MaxPool branch
        maxpool_out = self.maxpool(x)
        maxpool_out = self.conv_from_maxpool(maxpool_out)
        maxpool_out = self.maxpool_activation(maxpool_out)
        conv_outputs.append(maxpool_out)

        torchCat = _safe_import("torch.cat")
        x = torchCat(conv_outputs, dim=1)

        x = self.bn(x)
        x = self.output_activation(x)

        return x


class ShortcutLayer(NNModule):
    """Shortcut/Residual layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    activation : torch.nn.Module
        Activation function
    """

    def __init__(self, in_channels: int, out_channels: int, activation):
        super().__init__()

        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")

        self.conv = nnConv1d(
            in_channels, out_channels, kernel_size=1, padding=0, bias=False
        )
        self.bn = nnBatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, input_tensor, out_tensor):
        """Forward pass through shortcut layer.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor from the residual path
        out_tensor : torch.Tensor
            Output tensor from the main path

        Returns
        -------
        out : torch.Tensor
            Sum of transformed input and output, with activation
        """
        shortcut = self.conv(input_tensor)
        shortcut = self.bn(shortcut)

        x = shortcut + out_tensor
        x = self.activation(x)

        return x
