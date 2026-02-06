"""Convolutional Neural Network (CNN) for Classification and Regression in PyTorch."""

__all__ = ["CNNNetworkTorch"]

import numpy as np

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class CNNNetworkTorch(NNModule):
    """Establish the network structure for a CNN in PyTorch.

    Zhao et al. 2017 uses sigmoid activation in the hidden layers.
    To obtain same behaviour as Zhao et al. 2017, set activation_hidden to "sigmoid".

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of input (n_dims, series_length), i.e. (channels, length).
        Used to compute the flattened size after conv and pool layers.
    num_classes : int
        Number of output classes (classification) or 1 (regression).
    kernel_size : int, default = 7
        Length of the 1D convolution window.
    avg_pool_size : int, default = 3
        Size of the average pooling window.
    n_conv_layers : int, default = 2
        Number of convolutional plus average pooling layers.
    filter_sizes : array-like of int, shape = (n_conv_layers), default = None
        Number of filters per conv layer. If None, defaults to [6, 12].
    activation_hidden : str, default = "sigmoid"
        Activation function for hidden conv layers: "sigmoid" or "relu".
    activation : str or None, default = None
        Activation applied to the output layer. None for regression.
        For output layer, use None for regression (linear) or pass from
        classifier/regressor.
    use_bias : bool, default = True
        Whether to use a bias in fully connected layers.
    padding : str, default = "auto"
        Padding for conv layers. "auto": "same" if series_length < 60 else "valid";
        "valid" or "same" passed directly.
    random_state : int, default = 0
        Seed for reproducibility.

    References
    ----------
    .. [1] Zhao et al. Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics 28(1), 162--169, 2017
    """

    _tags = {
        "authors": ["hfawaz", "James-Large", "Withington", "TonyBagnall", "noxthot"],
        "maintainers": ["Faakhir30"],
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_shape,
        num_classes,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        filter_sizes=None,
        activation_hidden="sigmoid",
        activation=None,
        use_bias=True,
        padding="auto",
        random_state=0,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = filter_sizes
        self.activation_hidden = activation_hidden
        self.activation = activation
        self.use_bias = use_bias
        self.padding = padding
        self.random_state = random_state

        super().__init__()
        if filter_sizes is None:
            filter_sizes = [6, 12]
        fs = list(filter_sizes)
        # Extend to match n_conv_layers (same as TF)
        fs = fs[:n_conv_layers] + [fs[-1]] * max(0, n_conv_layers - len(fs))

        n_dims = input_shape[0]
        series_length = input_shape[1]

        if padding == "auto":
            padding_use = "same" if series_length < 60 else "valid"
        else:
            padding_use = padding

        padding_same = padding_use == "same"
        if padding_same:
            pad_value = kernel_size // 2
        else:
            pad_value = 0

        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        self.activation_hidden = activation_hidden
        self.activation = activation
        self.random_state = random_state

        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnAvgPool1d = _safe_import("torch.nn.AvgPool1d")
        nnLinear = _safe_import("torch.nn.Linear")
        nnFlatten = _safe_import("torch.nn.Flatten")

        # Build conv + pool blocks
        in_ch = n_dims
        layers = []
        for i in range(n_conv_layers):
            layers.append(
                nnConv1d(
                    in_ch,
                    fs[i],
                    kernel_size,
                    padding=pad_value,
                )
            )
            layers.append(self._instantiate_activation(activation_hidden))
            layers.append(nnAvgPool1d(avg_pool_size))
            in_ch = fs[i]

        self.conv_blocks = _safe_import("torch.nn.Sequential")(*layers)
        self.flatten = nnFlatten()

        # Compute flattened size
        L = series_length
        for _ in range(n_conv_layers):
            L_conv = L + 2 * pad_value - kernel_size + 1
            L = L_conv // avg_pool_size
        self._flattened_size = L * fs[-1]

        self.fc = nnLinear(self._flattened_size, num_classes, bias=use_bias)
        self._out_act = self._instantiate_activation(activation)

        if random_state is not None:
            torch_manual_seed = _safe_import("torch.manual_seed")
            torch_manual_seed(random_state)

    def forward(self, X):
        """Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Input of shape (batch_size, series_length, n_dims) from dataloader,
            or (batch_size, n_dims, series_length). We transpose if last dim
            equals n_dims and second is larger (assume batch, time, channels).

        Returns
        -------
        torch.Tensor
            Output of shape (batch_size, num_classes) or (batch_size, 1).
        """
        if isinstance(X, np.ndarray):
            torch_from_numpy = _safe_import("torch.from_numpy")
            X = torch_from_numpy(X).float()

        if X.dim() == 3:
            X = X.transpose(1, 2)  # (batch, n_dims, series_length)

        out = self.conv_blocks(X)
        out = self.flatten(out)
        out = self.fc(out)
        if self._out_act is not None:
            out = self._out_act(out)

        if self.num_classes == 1:  # (regression)
            out = out.squeeze(1)  # (batch_size,)
        return out

    def _instantiate_activation(self, activation):
        """Instantiate the activation function to be applied on the output layer.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied on the output layer.
        """
        # support for more activation functions will be added
        if activation is None or activation == "linear":
            return _safe_import("torch.nn.Identity")()
        elif isinstance(activation, NNModule):
            return activation
        elif isinstance(activation, str):
            if activation.lower() == "sigmoid":
                return _safe_import("torch.nn.Sigmoid")()
            elif activation.lower() == "relu":
                return _safe_import("torch.nn.ReLU")()
            elif activation.lower() == "softmax":
                return _safe_import("torch.nn.Softmax")(dim=1)
            elif activation.lower() == "logsoftmax":
                return _safe_import("torch.nn.LogSoftmax")(dim=1)
            elif activation.lower() == "logsigmoid":
                return _safe_import("torch.nn.LogSigmoid")()
            else:
                raise ValueError(
                    "If `activation` is not None, it must be one of "
                    "'sigmoid', 'logsigmoid', 'relu', 'softmax' or 'logsoftmax'. "
                    f"Found {activation}"
                )
        else:
            raise TypeError(
                "`activation` should either be of type str or torch.nn.Module. "
                f"But found the type to be: {type(activation)}"
            )
