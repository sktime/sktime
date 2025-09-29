"""Recurrent Neural Network (RNN) for Classification and Regression in PyTorch."""

__authors__ = ["RecreationalMath"]


import numpy as np

# from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")
nnRNN = _safe_import("torch.nn.RNN")
torch = _safe_import("torch")
nn = _safe_import("torch.nn")
nnDropout = _safe_import("torch.nn.Dropout")
nnIdentity = _safe_import("torch.nn.Identity")
nnLinear = _safe_import("torch.nn.Linear")
torchFrom_numpy = _safe_import("torch.from_numpy")
nnInitXavier_uniform_ = _safe_import("torch.nn.init.xavier_uniform_")
nnInitOrthogonal_ = _safe_import("torch.nn.init.orthogonal_")


class RNNNetworkTorch(NNModule):
    """Establish the network structure for an RNN in PyTorch.

    Parameters
    ----------
    input_size : int
        Number of expected features in the input
    hidden_dim : int, default = 6
        Number of features in the hidden state
    n_layers : int, default = 1
        Number of recurrent layers.
        E.g., setting n_layers=2 would mean stacking two RNNs together to form
        a stacked RNN, with the second RNN taking in outputs of the first RNN
        and computing the final results.
    activation : str/callable
        The activation function applied inside the RNN.
        Can be either 'tanh' or 'relu'. Default is 'relu'.
    bias : bool
        If False, then the layer does not use bias weights, default is True.
    batch_first : bool
        If True, then the input and output tensors are provided
        as (batch, seq, feature), default is False.
    num_classes : int
        Number of classes to predict.
    init_weights : bool
        If True, then the weights are initialized, default is True.
    dropout : float
        Dropout rate to apply. default is 0.0
    fc_dropout : float
        Dropout rate to apply to the fully connected layer. Default is 0.0
    bidirectional : bool
        If True, then the GRU is bidirectional, default is False.
    random_state   : int, default = 0
        seed to any needed random actions
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["RecreationalMath"],
        "maintainers": ["RecreationalMath"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 6,
        n_layers: int = 1,
        activation: str = "relu",
        bias: bool = False,
        batch_first: bool = False,
        num_classes: int = 2,
        init_weights: bool = True,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        bidirectional: bool = False,
        random_state: int = 0,
    ):
        self.random_state = random_state
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.n_layers = n_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        super().__init__()

        # Checking input dimensions.
        if isinstance(input_size, int):
            in_features = input_size
        elif isinstance(input_size, tuple):
            if len(input_size) == 3:
                in_features = input_size[1]
            else:
                raise ValueError(
                    "If `input_size` is a tuple, it must either be of length 3 and in "
                    "format (n_instances, n_dims, series_length). "
                    f"Found length of {len(input_size)}"
                )
        else:
            raise TypeError(
                "`input_size` should either be of type int or tuple. "
                f"But found the type to be: {type(input_size)}"
            )

        self.rnn = nnRNN(
            input_size=in_features,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            nonlinearity=self.activation,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        self.fc_dropout = fc_dropout
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.out_dropout = (
            nnDropout(p=self.fc_dropout) if self.fc_dropout else nnIdentity()
        )
        self.fc = nnLinear(
            self.hidden_dim * (2 if self.bidirectional else 1), self.num_classes
        )
        if self.init_weights:
            self.apply(self._init_weights)

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor of shape (seq_length, batch_size input_size)
            Input tensor containing the time series data.

        Returns
        -------
        out : torch.Tensor of shape (seq_length, batch_size, hidden_size)
            Output tensor containing the hidden states for each time step.
        """
        if isinstance(X, np.ndarray):
            X = torchFrom_numpy(X).float()
            # X = X.permute(1, 0, 2)
            # X = X.unsqueeze(0)

        out, _ = self.rnn(X)
        out = out[:, -1, :]
        out = self.out_dropout(out)
        out = self.fc(out)
        return out

    def _init_weights(self, module):
        """Apply Tensorflow like initializations.

        adapted from:
        https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization

        Parameters
        ----------
        module : torch.nn.Module
            Input module on which to apply initializations.
        """
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nnInitXavier_uniform_(param.data)
            elif "weight_hh" in name:
                nnInitOrthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
