"""Multi Channel Deep Convolution Neural Network (MCDCNN) in PyTorch."""

__authors__ = ["Faakhir30"]
__all__ = ["MCDCNNNetworkTorch"]


# from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class MCDCNNNetworkTorch(NNModule):
    """Multi Channel Deep Convolutional Neural Network (MCDCNN) in PyTorch.

    Adapted from the implementation of Fawaz et. al:
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mcdcnn.py

    Parameters
    ----------
    kernel_size : int, optional (default=5)
        The size of kernel in Conv1D layer.
    pool_size : int, optional (default=2)
        The size of kernel in (Max) Pool layer.
    filter_sizes : tuple, optional (default=(8, 8))
        The sizes of filter for Conv1D layer corresponding
        to each Conv1D in the block.
    dense_units : int, optional (default=732)
        The number of output units of the final Dense
        layer of this Network. This is NOT the final layer
        but the penultimate layer.
    conv_padding : str or None, optional (default="same")
        The type of padding to be applied to convolutional
        layers.
    pool_padding : str or None, optional (default="same")
        The type of padding to be applied to pooling layers.
    random_state : int, optional (default=0)
        The seed to any random action.
    activation : str or None, optional (default=None)
        The activation function to apply at the output.
        List of available activation functions:
        https://pytorch.org/docs/stable/nn.html#non-linear-activations-activation
    activation_hidden : string, default="relu"
        Activation function used in the hidden layers.
        List of available activation functions:
        https://pytorch.org/docs/stable/nn.html#non-linear-activations-activation
    use_bias : bool, optional (default=True)
        Whether bias should be included in the output layer.
    num_classes : int, optional (default=1)
        The number of classes to classify. Used for the output layer.
        Default is 1 (regression).
    """

    _tags = {
        "authors": ["hfawaz", "James-Large", "Withington", "noxthot"],
        "maintainers": ["Faakhir30"],
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        kernel_size=5,
        pool_size=2,
        filter_sizes=(8, 8),
        dense_units=732,
        conv_padding="same",
        pool_padding="same",
        random_state=0,
        activation=None,
        activation_hidden="relu",
        use_bias=True,
        num_classes=1,
    ):
        self.random_state = random_state
        self.activation = activation
        self.use_bias = use_bias
        self.activation_hidden = activation_hidden
        self.dense_units = dense_units
        self.num_classes = num_classes

        super().__init__()

        conv_pad_size = kernel_size // 2 if conv_padding == "same" else 0

        nnConv1d = _safe_import("torch.nn.Conv1d")

        self.conv1 = nnConv1d(
            in_channels=1,
            out_channels=filter_sizes[0],
            kernel_size=kernel_size,
            padding=conv_pad_size,
        )
        self.conv2 = nnConv1d(
            in_channels=filter_sizes[0],
            out_channels=filter_sizes[1],
            kernel_size=kernel_size,
            padding=conv_pad_size,
        )

        nnMaxPool1d = _safe_import("torch.nn.MaxPool1d")

        pool_pad_size = pool_size // 2 if pool_padding == "same" else 0
        self.pool = nnMaxPool1d(pool_size, padding=pool_pad_size)

        nnLinear = _safe_import("torch.nn.Linear")
        self.out = nnLinear(self.dense_units, self.num_classes, bias=self.use_bias)

        # Dense layer created lazily (depends on input length)
        self.fc = None

        self._activation_hidden = self._instantiate_activation(activation_hidden)
        if self.activation:
            self._activation = self._instantiate_activation(activation)

    def forward(self, X):
        """Forward pass of the MCDCNNNetworkTorch.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor of shape (batch_size, n_t, n_vars) from dataloader.
            The method transposes to (batch_size, n_vars, n_t) for Conv1d processing.
        """
        # Dataloader provides (batch_size, n_t, n_vars)
        # Transpose to (batch_size, n_vars, n_t) for Conv1d
        if X.dim() == 3:
            X = X.transpose(1, 2)  # (batch, n_t, n_vars) -> (batch, n_vars, n_t)
        batch_size, n_vars, n_t = X.shape
        outputs = []
        torchFlatten = _safe_import("torch.flatten")
        for i in range(n_vars):
            xi = X[:, i : i + 1, :]  # (batch, 1, n_t)

            x = self._activation_hidden(self.conv1(xi))
            x = self.pool(x)

            x = self._activation_hidden(self.conv2(x))
            x = self.pool(x)

            x = torchFlatten(x, start_dim=1)
            outputs.append(x)

        if n_vars == 1:
            x = outputs[0]
        else:
            nnCat = _safe_import("torch.cat")
            x = nnCat(outputs, dim=1)

        # Lazy dense creation
        if self.fc is None:
            nnLinear = _safe_import("torch.nn.Linear")
            self.fc = nnLinear(x.shape[1], self.dense_units, bias=self.use_bias).to(
                x.device
            )

        x = self._activation_hidden(self.fc(x))

        x = self.out(x)
        if self.activation:
            x = self._activation(x)

        # regression: (n_instances, 1) -> (n_instances,)
        if self.num_classes == 1:
            return x.squeeze(-1)

        return x

    def _instantiate_activation(self, activation):
        """Instantiate the activation function to be applied on the output layer.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied on the output layer.
        """
        if isinstance(activation, NNModule):
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
                    "'sigmoid', 'logsigmoid', 'softmax' or 'logsoftmax'. "
                    f"Found {activation}"
                )
        else:
            raise TypeError(
                "`activation` should either be of type str or torch.nn.Module. "
                f"But found the type to be: {type(activation)}"
            )
