"""Recurrent Neural Network (RNN) for Classification and Regression in PyTorch."""

__authors__ = ["RecreationalMath"]

import numpy as np

# from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class RNNNetworkTorch(NNModule):
    """Establish the network structure for an RNN in PyTorch.

    Parameters
    ----------
    input_size : int
        Number of expected features in the input
    num_classes : int
        Number of classes to predict
    hidden_dim : int, default = 6
        Number of features in the hidden state
    n_layers : int, default = 1
        Number of recurrent layers.
        E.g., setting n_layers=2 would mean stacking two RNNs together to form
        a stacked RNN, with the second RNN taking in outputs of the first RNN
        and computing the final results.
    activation : str or None, default = None
        Activation function used in the fully connected output layer. List of supported
        activation functions: ['sigmoid', 'softmax', 'logsoftmax', 'logsigmoid'].
        If None, then no activation function is applied.
    activation_hidden : str, default = "relu"
        The activation function applied inside the RNN. Can be either 'tanh' or 'relu'.
        Because currently PyTorch only supports these two activations inside the RNN.
        https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
    bias : bool, default = False
        If False, then the layer does not use bias weights.
    batch_first : bool, default = False
        If True, then the input and output tensors are provided
        as (batch, seq, feature) instead of (seq, batch, feature).
    init_weights : bool, default = True
        If True, then Tensorflow like initializations are applied to the weights.
        Adapted from:
        https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization
    dropout : float, default = 0.0
        If non-zero, introduces a Dropout layer on the outputs of each RNN layer except
        the fully connected output layer, with dropout probability equal to dropout.
    fc_dropout : float, default = 0.0
        If non-zero, introduces a Dropout layer on the outputs of the fully
        connected output layer, with dropout probability equal to fc_dropout.
    bidirectional : bool, default = False
        If True, then the RNN is bidirectional.
    random_state   : int, default = 0
        Seed to ensure reproducibility.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["RecreationalMath"],
        "maintainers": ["RecreationalMath"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_dim: int = 6,
        n_layers: int = 1,
        activation: str | None = None,
        activation_hidden: str = "relu",
        bias: bool = False,
        batch_first: bool = False,
        init_weights: bool = True,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        bidirectional: bool = False,
        random_state: int = 0,
    ):
        self.random_state = random_state
        self.hidden_dim = hidden_dim
        self.activation = activation
        # if activation_hidden is invalid, i.e. not in ['tanh', 'relu']
        # PyTorch will raise an error
        self.activation_hidden = activation_hidden
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

        nnRNN = _safe_import("torch.nn.RNN")
        self.rnn = nnRNN(
            input_size=in_features,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            nonlinearity=self.activation_hidden,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        self.fc_dropout = fc_dropout
        self.num_classes = num_classes
        self.init_weights = init_weights
        if self.fc_dropout:
            nnDropout = _safe_import("torch.nn.Dropout")
            self.out_dropout = nnDropout(p=self.fc_dropout)
        nnLinear = _safe_import("torch.nn.Linear")
        self.fc = nnLinear(
            in_features=self.hidden_dim * (2 if self.bidirectional else 1),
            out_features=self.num_classes,
        )
        # currently the above linear layer is only implemented for
        # SimpleRNNClassifierTorch, once RNNRegressorTorch is implemented,
        # changes will be made here
        # to handle the case when num_classes = 1 for regression
        if self.activation:
            self._activation = self._instantiate_activation()
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
            torchFrom_numpy = _safe_import("torch.from_numpy")
            X = torchFrom_numpy(X).float()
            # X = X.permute(1, 0, 2)
            # X = X.unsqueeze(0)

        out, _ = self.rnn(X)
        out = out[:, -1, :]
        if self.fc_dropout:
            out = self.out_dropout(out)
        out = self.fc(out)
        if self.activation:
            out = self._activation(out)
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
        nnInitXavier_uniform_ = _safe_import("torch.nn.init.xavier_uniform_")
        nnInitOrthogonal_ = _safe_import("torch.nn.init.orthogonal_")
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nnInitXavier_uniform_(param.data)
            elif "weight_hh" in name:
                nnInitOrthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def _instantiate_activation(self):
        """Instantiate the activation function to be applied on the output layer.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied on the output layer.
        """
        # support for more activation functions will be added
        # based on requirements of SimpleRNNRegressorTorch once it is implemented
        # currently only used in SimpleRNNClassifierTorch
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
