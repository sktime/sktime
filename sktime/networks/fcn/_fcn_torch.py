"""Fully Convolutional Network (FCN) for time series classification in PyTorch."""

__authors__ = ["Ali-John"]

import numpy as np

from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class FCNNetworkTorch(NNModule):
    """Establish the network structure for a FCN in PyTorch.

    Implements network in [1].

    Parameters
    ----------
    input_size : int
        Number of expected features in the input (number of dimensions/channels)
    num_classes : int
        Number of classes to predict
    n_conv_layers : int, default = 3
        Number of convolutional blocks in the network
    filter_sizes : list of int or None, default = None
        Number of filters for each convolutional layer. If None, defaults to
        [128, 256, 128] for 3 layers as specified in the original paper
    kernel_sizes : list of int or None, default = None
        Kernel sizes for each convolutional layer. If None, defaults to [8, 5, 3]
        for 3 layers as specified in the original paper
    activation : str or None or an instance of activation functions defined in
        torch.nn, default = None
        Activation function used in the fully connected output layer.
        List of supported activation functions: 'sigmoid', 'softmax',
        'logsoftmax', 'logsigmoid'. If None, no activation is applied.
    activation_hidden : str or None, default = "relu"
        Activation function used in the convolutional layers.
        List of supported activation functions: 'relu', 'tanh', 'sigmoid', etc.
    random_state : int, default = 0
        Seed to ensure reproducibility.

    References
    ----------
    .. [1] Network originally defined in:
    @inproceedings{wang2017time,
    title={Time series classification from scratch with deep neural networks:
    A strong baseline},
    author={Wang, Zhiguang and Yan, Weizhong and Oates, Tim},
    booktitle={2017 International joint conference on neural networks
    (IJCNN)},
    pages={1578--1585},
    year={2017},
    organization={IEEE}
    }
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Ali-John"],
        "maintainers": ["Ali-John"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        n_conv_layers: int = 3,
        filter_sizes: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        activation: str | None = None,
        activation_hidden: str = "relu",
        random_state: int = 0,
    ):
        self.num_classes = num_classes
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.random_state = random_state

        super().__init__()

        if isinstance(input_size, int):
            input_features = input_size
        elif isinstance(input_size, tuple):
            if len(input_size) == 3:
                input_features = input_size[1]
            else:
                raise ValueError(
                    "If `input_size` is a tuple, it must be of length 3 and in "
                    "format (n_instances, n_dims, series_length). "
                    f"Found length of `input_size` to be {len(input_size)}"
                )
        else:
            raise TypeError(
                "`input_size` should either be of type int or tuple. "
                f"But found the type to be: {type(input_size)}"
            )

        # Set defaults based on reference paper
        if self.filter_sizes is None:
            self.filter_sizes = [128, 256, 128][: self.n_conv_layers]
        if self.kernel_sizes is None:
            self.kernel_sizes = [8, 5, 3][: self.n_conv_layers]

        # Validate filter_sizes and kernel_sizes
        if len(self.filter_sizes) != self.n_conv_layers:
            raise ValueError(
                f"Length of filter_sizes ({len(self.filter_sizes)}) must match "
                f"n_conv_layers ({self.n_conv_layers})"
            )
        if len(self.kernel_sizes) != self.n_conv_layers:
            raise ValueError(
                f"Length of kernel_sizes ({len(self.kernel_sizes)}) must match "
                f"n_conv_layers ({self.n_conv_layers})"
            )

        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")
        nnLinear = _safe_import("torch.nn.Linear")
        nnSequential = _safe_import("torch.nn.Sequential")

        # Build convolutional layers container
        layers = []
        in_channels = input_features

        for i in range(self.n_conv_layers):
            out_channels = self.filter_sizes[i]
            kernel_size = self.kernel_sizes[i]

            layers.append(
                nnConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            # Add batch norm
            layers.append(nnBatchNorm1d(num_features=out_channels))
            # Add activation
            layers.append(self._instantiate_hidden_activation())

            in_channels = out_channels

        # Add adaptive average pooling layer
        layers.append(nnAdaptiveAvgPool1d(1))

        # Create the sequential container
        self.conv_layers = nnSequential(*layers)

        # fully connected layer
        self.fc = nnLinear(
            in_features=self.filter_sizes[-1], out_features=self.num_classes
        )

        # activation for output layer if provided
        if self.activation:
            self._activation = self._instantiate_activation()

    def forward(self, X):
        """Define the forward pass of the network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_length, input_size)
            Input data to the network

        Returns
        -------
        torch.Tensor of shape (batch_size, num_classes)
            Output of the network before the final classification layer
        """
        if isinstance(X, np.ndarray):
            torchFrom_numpy = _safe_import("torch.from_numpy")
            X = torchFrom_numpy(X).float()

        # PyTorch Conv1d expects input of shape (batch, channels, length)
        # DataLoader provides (batch, length, channels), so we transpose
        X = X.transpose(1, 2)

        # Pass through convolutional layers
        out = self.conv_layers(X)
        out = out.squeeze(-1)  # remove last dimension after global average pooling

        # fully connected layer
        out = self.fc(out)

        # apply output activation if specified
        if self.activation:
            out = self._activation(out)

        return out

    def _instantiate_hidden_activation(self):
        """Instantiate the activation function for hidden layers.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied in hidden layers.
        """
        if isinstance(self.activation_hidden, NNModule):
            return self.activation_hidden
        elif isinstance(self.activation_hidden, str):
            if self.activation_hidden.lower() == "relu":
                return _safe_import("torch.nn.ReLU")()
            elif self.activation_hidden.lower() == "tanh":
                return _safe_import("torch.nn.Tanh")()
            elif self.activation_hidden.lower() == "sigmoid":
                return _safe_import("torch.nn.Sigmoid")()
            elif self.activation_hidden.lower() == "leakyrelu":
                return _safe_import("torch.nn.LeakyReLU")()
            else:
                raise ValueError(
                    "If `activation_hidden` is not None, it must be one of "
                    "'relu', 'tanh', 'sigmoid', or 'leakyrelu'. "
                    f"Found {self.activation_hidden}"
                )
        else:
            raise TypeError(
                "`activation_hidden` should either be of type str or torch.nn.Module. "
                f"But found the type to be: {type(self.activation_hidden)}"
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
