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
    input_size : int
        Number of expected features in the input.
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
    init_weights: str or None, default = 'kaiming_uniform'
        The method to initialize the weights of the conv layers. Supported values are
        'kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', or None
        for default PyTorch initialization.
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
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        kernel_sizes: tuple = (8, 5, 3),
        filter_sizes: tuple = (128, 256, 128),
        lstm_size: int = 8,
        dropout: float = 0.8,
        attention: bool = False,
        activation: str | None = None,
        activation_hidden: str = "relu",
        init_weights: str | None = "kaiming_uniform",
        random_state: int = 0,
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
        self.init_weights = init_weights
        self.random_state = random_state

        super().__init__()

        # LSTM Arm
        if self.attention:
            # Attention mechanism: Multi-head attention
            nnMultiheadAttention = _safe_import("torch.nn.MultiheadAttention")
            self.attention_layer = nnMultiheadAttention(
                embed_dim=self.input_size, num_heads=1, batch_first=True
            )

        nnLSTM = _safe_import("torch.nn.LSTM")

        self.lstm = nnLSTM(
            input_size=self.input_size,
            hidden_size=self.lstm_size,
            num_layers=1,
            batch_first=True,
        )

        nnDropout = _safe_import("torch.nn.Dropout")
        self.lstm_dropout = nnDropout(p=self.dropout)

        # CNN Arm (Fully Convolutional Network)
        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")

        nnModuleList = _safe_import("torch.nn.ModuleList")
        self.conv_layers = nnModuleList()
        self.bn_layers = nnModuleList()
        for i in range(len(self.filter_sizes)):
            in_channels = self.input_size if i == 0 else self.filter_sizes[i - 1]
            out_channels = self.filter_sizes[i]
            conv_layer = nnConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_sizes[i],
                padding="same",
            )
            bn_layer = nnBatchNorm1d(out_channels)
            self.conv_layers.append(conv_layer)
            self.bn_layers.append(bn_layer)

        # Global Average Pooling
        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")
        self.global_avg_pool = nnAdaptiveAvgPool1d(1)

        self._activation_hidden = self._instantiate_activation(self.activation_hidden)
        if self.activation is not None:
            self._activation = self._instantiate_activation(self.activation)

        # Output layer (concatenated LSTM + CNN outputs)
        # LSTM outputs lstm_size, CNN outputs filter_sizes[-1]
        nnLinear = _safe_import("torch.nn.Linear")
        self.fc = nnLinear(
            in_features=self.lstm_size + self.filter_sizes[-1],
            out_features=self.num_classes,
        )

        if self.init_weights is not None:
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
        y = x_cnn
        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            y = conv_layer(y)
            y = bn_layer(y)
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

            if module.bias is not None:
                module.bias.data.zero_()

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
