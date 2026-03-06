"""Fully Convolutional Network (FCN) for Classification and Regression in PyTorch."""

__authors__ = ["kajal-jotwani"]
__all__ = ["FCNNetworkTorch"]

import numpy as np

from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class FCNNetworkTorch(NNModule):
    """Fully Convolutional Network (FCN) for classification and regression in PyTorch.

    Parameters
    ----------
    input_size : int
        Number of input features (channels / n_dims).
    num_classes : int
        Number of output classes (for classification) or 1 (for regression).
    filter_sizes : tuple of int, default = (128, 256, 128)
        Number of filters for each convolutional layer. The number of
        convolutional blocks is inferred from the length of this tuple.
    kernel_sizes : tuple of int, default = (8, 5, 3)
        Kernel size for each convolutional layer. Must have the same length
        as ``filter_sizes``.
    activation_hidden : str, default = "relu"
        Activation function applied after each BatchNorm layer in the
        convolutional blocks. Supported: ``"relu"``, ``"tanh"``, ``"sigmoid"``.
    activation : str or None, default = None
        Activation function used in the fully connected output layer.
        Supported: ``"sigmoid"``, ``"softmax"``, ``"logsoftmax"``,
        ``"logsigmoid"``, or ``None`` for no activation.
    init_weights : str or None, default = None
        The method to initialize the weights of the convolutional layers.
        Supported: ``"kaiming_uniform"``, ``"kaiming_normal"``,
        ``"xavier_uniform"``, ``"xavier_normal"``, or ``None`` for default
        PyTorch initialization.
        The TensorFlow FCN does not set any explicit initializer
        (it relies on Keras defaults), so ``None`` is used here to let
        PyTorch use its own defaults as well.
    random_state : int, default = 0
        Seed for reproducibility.

    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        filter_sizes: tuple = (128, 256, 128),
        kernel_sizes: tuple = (8, 5, 3),
        activation_hidden: str = "relu",
        activation: str | None = None,
        init_weights: str | None = None,
        random_state: int = 0,
    ):
        if len(filter_sizes) != len(kernel_sizes):
            raise ValueError(
                "The length of `filter_sizes` must equal the length of "
                f"`kernel_sizes`. Got filter_sizes={filter_sizes} "
                f"(length {len(filter_sizes)}) and kernel_sizes={kernel_sizes} "
                f"(length {len(kernel_sizes)})."
            )

        self.input_size = input_size
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.activation_hidden = activation_hidden
        self.activation = activation
        self.init_weights = init_weights
        self.random_state = random_state
        super().__init__()

        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnModuleList = _safe_import("torch.nn.ModuleList")

        # Build convolutional blocks dynamically based on filter_sizes length.
        # Each block: Conv1d -> BatchNorm1d -> activation_hidden
        n_layers = len(self.filter_sizes)
        self.conv_layers = nnModuleList()
        self.bn_layers = nnModuleList()

        in_channels = self.input_size
        for i in range(n_layers):
            self.conv_layers.append(
                nnConv1d(
                    in_channels=in_channels,
                    out_channels=self.filter_sizes[i],
                    kernel_size=self.kernel_sizes[i],
                    padding="same",
                )
            )
            self.bn_layers.append(nnBatchNorm1d(self.filter_sizes[i]))
            in_channels = self.filter_sizes[i]

        # Global Average Pooling collapses the time dimension
        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")
        self.gap = nnAdaptiveAvgPool1d(1)

        # Fully connected output layer
        nnLinear = _safe_import("torch.nn.Linear")
        self.fc = nnLinear(
            in_features=self.filter_sizes[-1],
            out_features=self.num_classes,
        )

        # Hidden activation (used inside conv blocks)
        self._hidden_activation = self._get_hidden_activation()

        # Output activation (used after FC layer)
        if self.activation:
            self._output_activation = self._instantiate_activation()

        # Weight initialization only applied when init_weights is not None
        if self.init_weights is not None:
            self.apply(self._init_weights)

    def forward(self, X):
        """Forward pass through the FCN network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_length, n_dims)
            Input tensor containing the time series data.

        Returns
        -------
        out : torch.Tensor of shape (batch_size, num_classes)
            Output tensor containing the predictions.
        """
        if isinstance(X, np.ndarray):
            torchFrom_numpy = _safe_import("torch.from_numpy")
            X = torchFrom_numpy(X).float()

        # Conv1d expects (batch, channels, length) format
        x = X.transpose(1, 2)  # (batch, n_dims, seq_length)

        # Pass through each conv block: Conv1d -> BatchNorm -> activation
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = self._hidden_activation(x)

        # Global Average Pooling
        # (batch, filter_sizes[-1], seq_length) -> (batch, filter_sizes[-1], 1)
        x = self.gap(x)
        x = x.squeeze(-1) 

        # Fully connected output layer
        out = self.fc(x)

        if self.activation:
            out = self._output_activation(out)

        return out

    def _get_hidden_activation(self):
        """Return the activation function for the convolutional blocks.

        Returns
        -------
        activation : torch.nn.Module
            The activation function instance.
        """
        activation_name = self.activation_hidden.lower()
        if activation_name == "relu":
            return _safe_import("torch.nn.ReLU")()
        elif activation_name == "tanh":
            return _safe_import("torch.nn.Tanh")()
        elif activation_name == "sigmoid":
            return _safe_import("torch.nn.Sigmoid")()
        else:
            raise ValueError(
                f"Unsupported hidden activation: {self.activation_hidden}. "
                "Supported values: 'relu', 'tanh', 'sigmoid'."
            )

    def _instantiate_activation(self):
        """Instantiate the activation function for the output layer.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied on the output layer.
        """
        if isinstance(self.activation, NNModule):
            return self.activation
        elif isinstance(self.activation, str):
            if self.activation.lower() == "sigmoid":
                return _safe_import("torch.nn.Sigmoid")()
            elif self.activation.lower() == "softmax":
                return _safe_import("torch.nn.Softmax")(dim=1)
            elif self.activation.lower() == "logsoftmax":
                return _safe_import("torch.nn.LogSoftmax")(dim=1)
            elif self.activation.lower() == "logsigmoid":
                return _safe_import("torch.nn.LogSigmoid")()
            else:
                raise ValueError(
                    "If `activation` is not None, it must be one of "
                    "'sigmoid', 'logsigmoid', 'softmax' or 'logsoftmax'. "
                    f"Found {self.activation}"
                )
        else:
            raise TypeError(
                "`activation` should either be of type str or torch.nn.Module. "
                f"But found the type to be: {type(self.activation)}"
            )

    def _init_weights(self, module):
        """Initialize weights of convolutional layers.

        Parameters
        ----------
        module : torch.nn.Module
            Input module on which to apply initializations.
        """
        nnConv1d = _safe_import("torch.nn.Conv1d")

        kaiming_uniform_ = _safe_import("torch.nn.init.kaiming_uniform_")
        kaiming_normal_ = _safe_import("torch.nn.init.kaiming_normal_")
        xavier_uniform_ = _safe_import("torch.nn.init.xavier_uniform_")
        xavier_normal_ = _safe_import("torch.nn.init.xavier_normal_")

        if isinstance(module, nnConv1d):
            if self.init_weights == "kaiming_uniform":
                kaiming_uniform_(module.weight, nonlinearity="relu")
            elif self.init_weights == "kaiming_normal":
                kaiming_normal_(module.weight, nonlinearity="relu")
            elif self.init_weights == "xavier_uniform":
                xavier_uniform_(module.weight)
            elif self.init_weights == "xavier_normal":
                xavier_normal_(module.weight)
            else:
                raise ValueError(
                    f"Unsupported init_weights method: {self.init_weights}. "
                    "Supported: 'kaiming_uniform', 'kaiming_normal', "
                    "'xavier_uniform', 'xavier_normal', or None."
                )
            if module.bias is not None:
                module.bias.data.zero_()