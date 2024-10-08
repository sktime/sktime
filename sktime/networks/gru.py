"""Gated Recurrent Unit (GRU) for time series classification."""

__author__ = ["fnhirwa"]

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn

    NNModule = nn.Module
    NNSequential = nn.Sequential
else:

    class NNModule:
        """Dummy class if torch is unavailable."""

    class NNSequential:
        """Dummy class if torch is unavailable."""


class GRU(NNModule):
    """Gated Recurrent Unit (GRU) for time series classification.

    Network originally defined in [1]_, [2]_ and [3]_.

    Parameters
    ----------
    input_size : int
        Number of features in the input time series.
    hidden_dim : int
        Number of features in the hidden state.
    n_layers : int
        Number of recurrent layers.
    batch_first : bool
        If True, then the input and output tensors are provided
        as (batch, seq, feature), default is False.
    bias : bool
        If False, then the layer does not use bias weights, default is True.
    num_classes : int
        Number of classes to predict.
    init_weights : bool
        If True, then the weights are initialized, default is True.
    dropout : float
        Dropout rate to apply. default is 0.0
    fc_dropout : float
        Dropout rate to apply to the fully connected layer. default is 0.0
    bidirectional : bool
        If True, then the GRU is bidirectional, default is False.

    References
    ----------

    .. [1] Cho, Kyunghyun, et al. "Learning phrase representations
        using RNN encoder-decoder for statistical machine translation."
        arXiv preprint arXiv:1406.1078 (2014).
    .. [2] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio.
        Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.
        arXiv preprint arXiv:1412.3555 (2014).
    .. [3] https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    """

    _tags = {
        # packaging info
        # --------------
        "author": ["fnhirwa"],
        "maintainers": ["fnhirwa"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
    }

    def __init__(
        self: "GRU",
        input_size: int,
        hidden_dim: int,
        n_layers: int,
        batch_first=False,
        bias=True,
        num_classes=None,
        init_weights=True,
        dropout=0.0,
        fc_dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.init_weights = init_weights
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=batch_first,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.fc_dropout = fc_dropout
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.out_dropout = (
            nn.Dropout(self.fc_dropout) if self.fc_dropout else nn.Identity()
        )
        self.fc = nn.Linear(
            self.hidden_dim * (1 + self.bidirectional), self.num_classes
        )
        if self.init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        # adapted from https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization #noqa: E501
        # Tensoflow like initialization
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        X, _ = self.gru(X)
        X = X[:, -1, :]
        X = self.out_dropout(X)
        X = self.fc(X)
        return X


class GRUFCNN(NNModule):
    """GRU with Convolutional Neural Network (FCNN) for time series classification.

    This network was originally defined in [1]_.
    The current implementation uses PyTorch and references
    the TensorFlow implementations in [2]_ and [3]_.

    Parameters
    ----------
    input_size : int
        Number of features in the input time series.
    hidden_dim : int
        Number of features in the hidden state.
    gru_layers : int
        Number of GRU layers.
    batch_first : bool
        If True, then the input and output tensors are provided
        as (batch, seq, feature), default is False.
    bias : bool
        If False, then the layer does not use bias weights, default is True.
    num_classes : int
        Number of classes to predict.
    init_weights : bool
        If True, then the weights are initialized, default is True.
    dropout : float
        Dropout rate to apply inside gru cell. default is 0.0
    gru_dropout : float
        Dropout rate to apply to the gru output layer. default is 0.0
    bidirectional : bool
        If True, then the GRU is bidirectional, default is False.
    conv_layers : list
        List of integers specifying the number of filters in each convolutional layer.
        default is [128, 256, 128].
    kernel_sizes : list
        List of integers specifying the kernel size in each convolutional layer.
        default is [7, 5, 3].

    References
    ----------
    .. [1] Elsayed, et al. "Deep Gated Recurrent and Convolutional Network Hybrid Model
        for Univariate Time Series Classification."
        arXiv preprint arXiv:1812.07683 (2018).
    .. [2] https://github.com/NellyElsayed/GRU-FCN-model-for-univariate-time-series-classification
    .. [3] https://github.com/titu1994/LSTM-FCN

    """

    _tags = {
        # packaging info
        # --------------
        "author": ["fnhirwa"],
        "maintainers": ["fnhirwa"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
    }

    def __init__(
        self: "GRUFCNN",
        input_size: int,
        hidden_dim: int,
        gru_layers: int,
        batch_first: bool = False,
        bias: bool = True,
        num_classes: int = None,
        init_weights: bool = True,
        dropout: float = 0.0,
        gru_dropout: float = 0.0,
        bidirectional: bool = False,
        conv_layers: list = [128, 256, 128],
        kernel_sizes: list = [7, 5, 3],
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.init_weights = init_weights
        self.dropout = dropout
        self.bidirectional = bidirectional
        # gru
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=batch_first,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.gru_dropout = gru_dropout
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.conv_layers = conv_layers
        self.kernel_sizes = kernel_sizes

        # fcn
        self.permute = Permute(0, 2, 1)
        self.conv1 = Conv(
            in_channels=input_size,
            out_channels=conv_layers[0],
            kernel_size=kernel_sizes[0],
        )
        self.conv2 = Conv(
            in_channels=conv_layers[0],
            out_channels=conv_layers[1],
            kernel_size=kernel_sizes[1],
        )
        self.conv3 = Conv(
            in_channels=conv_layers[1],
            out_channels=conv_layers[2],
            kernel_size=kernel_sizes[2],
        )
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)

        # combined
        self.concat = Concat()
        self.grudropout = (
            nn.Dropout(self.gru_dropout) if self.gru_dropout else nn.Identity()
        )
        self.fc = nn.Linear(
            hidden_dim * (1 + bidirectional) + conv_layers[-1], num_classes
        )

        # weights initialization
        if self.init_weights:
            self.apply(self._init_gru_weights)
            self.apply(self._init_conv_weights)

    def _init_gru_weights(self, module):
        # adapted from https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization #noqa: E501
        # Tensoflow like initialization
        # this initialization is only GlorotUniform means xavier_uniform
        for name, param in module.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def _init_conv_weights(self, module):
        # the initialization is based on the original paper
        for name, param in module.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.kaiming_normal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        # GRU
        gru_out, _ = self.gru(X)
        gru_out = gru_out[:, -1, :]
        gru_out = self.grudropout(gru_out)  # apply dropout

        # FCN
        X = self.permute(X)
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.globalavgpool(X)
        X = X.view(X.size(0), -1)

        # Concatenate
        X = self.concat(gru_out, X)
        X = self.fc(X)
        return X


class Conv(NNSequential):
    """Convolutional Block for FCN.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the kernel.
    padding : str
        Padding type, default is 'same'.
    """

    def __init__(
        self: "Conv",
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding="same",
    ):
        super().__init__(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )


class Concat(NNModule):
    """Concatenation of two tensors.

    Parameters
    ----------
    dim : int
        Dimension along which to concatenate.
    """

    def __init__(self: "Concat", dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        """Forward pass through the network.

        Parameters
        ----------
        x1 : torch.Tensor
            Input tensor 1.
        x2 : torch.Tensor
            Input tensor 2.
        """
        return torch.cat((x1, x2), dim=self.dim)


class Permute(NNModule):
    """Permute the dimensions of a tensor.

    Parameters
    ----------
    dims : tuple
        New order of dimensions.
    """

    def __init__(self: "Permute", *dims: tuple):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        return x.permute(self.dims)
