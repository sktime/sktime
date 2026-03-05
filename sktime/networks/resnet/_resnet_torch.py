"""Residual Network (ResNet) for classification and regression in PyTorch."""

__authors__ = ["DCchoudhury15"]
__all__ = ["ResNetNetworkTorch"]

import numpy as np

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class ResNetNetworkTorch(NNModule):
    """Establish the network structure for a ResNet in PyTorch.

    Adapted from the TensorFlow implementation in sktime which itself was adapted
    from https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    Parameters
    ----------
    input_size : int
        Number of input channels (number of time series dimensions).
    num_classes : int
        Number of output neurons. For classification, this is the number
        of classes. For regression, this is 1.
    n_filters_per_block : tuple of int, default = (64, 128, 128)
        Number of convolutional filters in each residual block.
        The length of this tuple determines the number of residual blocks.
    kernel_sizes : tuple of int, default = (8, 5, 3)
        Kernel sizes of the 3 Conv1d layers within each residual block.
        Must be a tuple of length 3.
    padding : str, default = "same"
        Padding mode for all Conv1d layers.
    activation_hidden : str, default = "relu"
        Activation function used after batch normalization in the residual blocks.
        Supported values: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu',
        'selu', 'gelu'.
    activation : str or None, default = None
        Activation function applied to the output layer.
        If None, no activation is applied (raw logits are returned).
        Supported values: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu',
        'selu', 'gelu', 'softmax', 'logsoftmax', 'logsigmoid'.
    use_bias : bool, default = True
        Whether the final fully connected layer uses a bias vector.
    init_weights : bool, default = True
        Whether to apply weight initialization to Conv1d layers.
        If True, applies ``kaiming_uniform_`` initialization
        (appropriate for ReLU-based convolutional networks).
    random_state : int, default = 0
        Seed to ensure reproducibility.

    References
    ----------
    .. [1] Wang et al, Time series classification from scratch with deep
        neural networks: A strong baseline, IJCNN 2017.
    """

    _tags = {
        "authors": ["DCchoudhury15"],
        "maintainers": ["DCchoudhury15"],
        "python_version": ">=3.10,<3.15",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        n_filters_per_block: tuple = (64, 128, 128),
        kernel_sizes: tuple = (8, 5, 3),
        padding: str = "same",
        activation_hidden: str = "relu",
        activation: str | None = None,
        use_bias: bool = True,
        init_weights: bool = True,
        random_state: int = 0,
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.n_filters_per_block = n_filters_per_block
        self.kernel_sizes = kernel_sizes
        self.padding = padding
        self.activation_hidden = activation_hidden
        self.activation = activation
        self.use_bias = use_bias
        self.init_weights = init_weights
        self.random_state = random_state

        super().__init__()

        if len(self.kernel_sizes) != 3:
            raise ValueError(
                "ResNet residual blocks always contain exactly 3 Conv1d layers. "
                f"`kernel_sizes` must be a tuple of length 3, "
                f"but got length {len(self.kernel_sizes)}."
            )

        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnModuleList = _safe_import("torch.nn.ModuleList")
        nnIdentity = _safe_import("torch.nn.Identity")

        self.block_conv_layers = nnModuleList()
        self.block_bn_layers = nnModuleList()
        self.shortcut_convs = nnModuleList()
        self.shortcut_bns = nnModuleList()

        for block_idx in range(len(self.n_filters_per_block)):
            n_filters = self.n_filters_per_block[block_idx]
            if block_idx == 0:
                in_channels = self.input_size
            else:
                in_channels = self.n_filters_per_block[block_idx - 1]

            conv_layers = nnModuleList()
            bn_layers = nnModuleList()
            for conv_idx in range(3):
                conv_in = in_channels if conv_idx == 0 else n_filters
                conv_layers.append(
                    nnConv1d(
                        in_channels=conv_in,
                        out_channels=n_filters,
                        kernel_size=self.kernel_sizes[conv_idx],
                        padding=self.padding,
                    )
                )
                bn_layers.append(nnBatchNorm1d(n_filters))

            self.block_conv_layers.append(conv_layers)
            self.block_bn_layers.append(bn_layers)

            if in_channels != n_filters:
                self.shortcut_convs.append(
                    nnConv1d(
                        in_channels=in_channels,
                        out_channels=n_filters,
                        kernel_size=1,
                    )
                )
            else:
                self.shortcut_convs.append(nnIdentity())

            self.shortcut_bns.append(nnBatchNorm1d(n_filters))

        self._activation_hidden = self._instantiate_activation(self.activation_hidden)
        self.gap = _safe_import("torch.nn.AdaptiveAvgPool1d")(1)

        nnLinear = _safe_import("torch.nn.Linear")
        last_n_filters = self.n_filters_per_block[-1]
        self.fc = nnLinear(last_n_filters, self.num_classes, bias=self.use_bias)

        if self.activation is not None:
            self._activation = self._instantiate_activation(self.activation)

        if self.init_weights:
            self.apply(self._init_weights)

    def forward(self, X):
        """Forward pass through the ResNet network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_length, n_dims)
            Input tensor containing the time series data.

        Returns
        -------
        out : torch.Tensor of shape (batch_size, num_classes)
            Output tensor containing the logits or activated outputs.
        """
        if isinstance(X, np.ndarray):
            torchFrom_numpy = _safe_import("torch.from_numpy")
            X = torchFrom_numpy(X).float()

        out = X.transpose(1, 2)

        for block_idx in range(len(self.n_filters_per_block)):
            conv_layers = self.block_conv_layers[block_idx]
            bn_layers = self.block_bn_layers[block_idx]
            shortcut_conv = self.shortcut_convs[block_idx]
            shortcut_bn = self.shortcut_bns[block_idx]

            main = self._activation_hidden(bn_layers[0](conv_layers[0](out)))
            main = self._activation_hidden(bn_layers[1](conv_layers[1](main)))
            main = bn_layers[2](conv_layers[2](main))

            shortcut = shortcut_bn(shortcut_conv(out))
            out = self._activation_hidden(main + shortcut)

        out = self.gap(out).squeeze(-1)
        out = self.fc(out)

        if self.activation is not None:
            out = self._activation(out)

        return out

    def _init_weights(self, module):
        """Apply kaiming_uniform initialization to Conv1d modules.

        Parameters
        ----------
        module : torch.nn.Module
            Input module on which to apply initializations.
        """
        nnConv1d = _safe_import("torch.nn.Conv1d")
        if isinstance(module, nnConv1d):
            kaiming_uniform_ = _safe_import("torch.nn.init.kaiming_uniform_")
            kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.zero_()

    def _instantiate_activation(self, activation):
        """Instantiate an activation function.

        Parameters
        ----------
        activation : str
            Name of the activation function.

        Returns
        -------
        activation_function : torch.nn.Module
            The instantiated activation function.
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
            elif act == "softmax":
                return _safe_import("torch.nn.Softmax")(dim=1)
            elif act == "logsoftmax":
                return _safe_import("torch.nn.LogSoftmax")(dim=1)
            elif act == "logsigmoid":
                return _safe_import("torch.nn.LogSigmoid")()
            else:
                raise ValueError(
                    "If `activation` is a string, it must be one of "
                    "'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', "
                    "'gelu', 'softmax', 'logsoftmax', or 'logsigmoid'. "
                    f"Found '{activation}'"
                )
        else:
            raise TypeError(
                "`activation` should either be of type str or torch.nn.Module. "
                f"But found the type to be: {type(activation)}"
            )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {
            "input_size": 1,
            "num_classes": 2,
        }
        params2 = {
            "input_size": 2,
            "num_classes": 3,
            "n_filters_per_block": (16, 32),
            "kernel_sizes": (3, 3, 3),
            "random_state": 42,
        }
        return [params1, params2]
