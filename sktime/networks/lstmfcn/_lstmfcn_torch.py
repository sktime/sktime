"""LongShort Term Memory Fully Convolutional Network (LSTM-FCN) in PyTorch."""

__authors__ = ["Faakhir30"]
__all__ = ["LSTMFCNNetworkTorch"]

import numpy as np

# from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class LSTMFCNNetworkTorch(NNModule):
    """Implementation of LSTM-FCN network from Karim et al (2019) [1] in PyTorch.

    Overview
    --------
    Combines an LSTM arm with a CNN arm. Optionally uses an attention mechanism in the
    LSTM which the author indicates provides improved performance.

    Parameters
    ----------
    input_size : int or tuple
        Number of expected features in the input. Can be an int representing the
        number of input features, or a tuple of shape (n_instances, n_dims, series_len).
    num_classes : int
        Number of classes to predict
    kernel_sizes : tuple of int, default=(8, 5, 3)
        Specifying the length of the 1D convolution windows for each conv layer
    filter_sizes : tuple of int, default=(128, 256, 128)
        Size of filter for each conv layer
    lstm_size : int, default=8
        Output dimension for LSTM layer (hidden state size)
    dropout : float, default=0.8
        Controls dropout rate of LSTM layer
    attention : bool, default=False
        If True, uses attention mechanism before LSTM layer
    activation : str or None, default=None
        Activation function used in the output layer.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu'
    activation_hidden : str, default="relu"
        Activation function used for convolutional layers.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu'
    random_state : int, default=0
        Seed to ensure reproducibility

    Notes
    -----
    Ported from TensorFlow implementation in sktime
    Based on the paper by Karim et al (2019)

    References
    ----------
    .. [1] Karim et al. Multivariate LSTM-FCNs for Time Series Classification, 2019
    https://arxiv.org/pdf/1801.04503.pdf
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Faakhir30"],
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
        kernel_sizes: tuple = (8, 5, 3),
        filter_sizes: tuple = (128, 256, 128),
        random_state: int = 0,
        lstm_size: int = 8,
        dropout: float = 0.8,
        attention: bool = False,
        activation: str | None = None,
        activation_hidden: str = "relu",
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes
        self.filter_sizes = filter_sizes
        self.lstm_size = lstm_size
        self.dropout = dropout
        self.attention = attention
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.random_state = random_state

        super().__init__()

        # Checking input dimensions
        if isinstance(self.input_size, int):
            in_features = self.input_size
        elif isinstance(self.input_size, tuple):
            if len(self.input_size) == 3:
                in_features = self.input_size[1]
            else:
                raise ValueError(
                    "If `input_size` is a tuple, it must be of length 3 and in "
                    "format (n_instances, n_dims, series_length). "
                    f"Found length of {len(self.input_size)}"
                )
        else:
            raise TypeError(
                "`input_size` should either be of type int or tuple. "
                f"But found the type to be: {type(self.input_size)}"
            )

        # LSTM Arm
        if self.attention:
            # Attention mechanism: Multi-head attention
            nnMultiheadAttention = _safe_import("torch.nn.MultiheadAttention")
            self.attention_layer = nnMultiheadAttention(
                embed_dim=in_features, num_heads=1, batch_first=True
            )

        nnLSTM = _safe_import("torch.nn.LSTM")

        self.lstm = nnLSTM(
            input_size=in_features,
            hidden_size=self.lstm_size,
            num_layers=1,
            batch_first=True,
        )

        nnDropout = _safe_import("torch.nn.Dropout")
        self.lstm_dropout = nnDropout(p=self.dropout)

        # CNN Arm (Fully Convolutional Network)
        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")

        # First Conv Block
        self.conv1 = nnConv1d(
            in_channels=in_features,
            out_channels=self.filter_sizes[0],
            kernel_size=self.kernel_sizes[0],
            padding="same",
        )
        self.bn1 = nnBatchNorm1d(self.filter_sizes[0])

        # Second Conv Block
        self.conv2 = nnConv1d(
            in_channels=self.filter_sizes[0],
            out_channels=self.filter_sizes[1],
            kernel_size=self.kernel_sizes[1],
            padding="same",
        )
        self.bn2 = nnBatchNorm1d(self.filter_sizes[1])

        # Third Conv Block
        self.conv3 = nnConv1d(
            in_channels=self.filter_sizes[1],
            out_channels=self.filter_sizes[2],
            kernel_size=self.kernel_sizes[2],
            padding="same",
        )
        self.bn3 = nnBatchNorm1d(self.filter_sizes[2])

        # Global Average Pooling
        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")
        self.global_avg_pool = nnAdaptiveAvgPool1d(1)

        self._activation_hidden = self._instantiate_activation(self.activation_hidden)
        if self.activation is not None:
            self._activation = self._instantiate_activation(self.activation)

        # Output layer (concatenated LSTM + CNN outputs)
        # LSTM outputs lstm_size, CNN outputs filter_sizes[2]
        nnLinear = _safe_import("torch.nn.Linear")
        self.fc = nnLinear(
            in_features=self.lstm_size + self.filter_sizes[2],
            out_features=self.num_classes,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, X):
        """Forward pass through the LSTM-FCN network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_length, n_dims)
            Input tensor containing the time series data.

        Returns
        -------
        out : torch.Tensor of shape (batch_size, num_classes)
            Output tensor containing the class predictions.
        """
        if isinstance(X, np.ndarray):
            torchFrom_numpy = _safe_import("torch.from_numpy")
            X = torchFrom_numpy(X).float()

        # LSTM Arm
        x_lstm = X

        if self.attention:
            x_lstm, _ = self.attention_layer(x_lstm, x_lstm, x_lstm)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x_lstm)
        # Extract the last hidden state
        # h_n shape: (num_layers, batch, hidden_size)
        lstm_out = h_n[-1]  # (batch, lstm_size)
        lstm_out = self.lstm_dropout(lstm_out)

        # CNN Arm
        # Conv1d expects (batch, channels, length) format
        x_cnn = X.transpose(1, 2)  # (batch, n_dims, seq_length)
        y = self.conv1(x_cnn)
        y = self.bn1(y)
        y = self._activation_hidden(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self._activation_hidden(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self._activation_hidden(y)

        # Global Average Pooling
        y = self.global_avg_pool(y)  # (batch, filter_sizes[2], 1)
        y = y.squeeze(-1)  # (batch, filter_sizes[2])

        # Concatenate LSTM and CNN outputs
        torchCat = _safe_import("torch.cat")
        concatenated = torchCat([lstm_out, y], dim=1)

        # Final output layer
        out = self.fc(concatenated)
        if self.activation is not None:
            out = self._activation(out)

        # Squeeze if regression
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

        if isinstance(module, nnConv1d):
            # He uniform for Conv layers
            nnInitKaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.fill_(0)

    def _instantiate_activation(self, activation):
        """Instantiate the activation function for CNN layers.

        Parameters
        ----------
        activation : str
            Activation function to instantiate.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied in the CNN arm.
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
                    "'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', or 'gelu'."
                    f"Found {activation}"
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
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            # Advanced model version with attention
            {
                "input_size": 10,
                "num_classes": 2,
                "kernel_sizes": (8, 5, 3),
                "filter_sizes": (128, 256, 128),
                "lstm_size": 8,
                "dropout": 0.25,
                "attention": True,
            },
            # Simpler model version without attention
            {
                "input_size": 10,
                "num_classes": 2,
                "kernel_sizes": (4, 2, 1),
                "filter_sizes": (32, 64, 32),
                "lstm_size": 8,
                "dropout": 0.75,
                "attention": False,
            },
            # Minimal configuration
            {
                "input_size": 5,
                "num_classes": 3,
            },
        ]

        return params
