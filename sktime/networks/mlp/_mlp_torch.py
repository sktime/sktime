"""MLP (Multi-Layer Perceptron) deep learning network structure in PyTorch.

For Classification and Regression.
"""

__authors__ = ["RecreationalMath"]
__all__ = ["MLPNetworkTorch"]


from collections.abc import Callable

import numpy as np

from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class MLPNetworkTorch(NNModule):
    """Establish the network structure for an MLP in PyTorch.

    Implements a simple MLP network, as in [1]_.

    Parameters
    ----------
    input_shape : tuple
        shape of the input data fed into the network
    num_classes : int
        Number of classes to predict
    hidden_dim : int, default = 500
        Number of features in the hidden state
    n_layers : int, default = 4
        Number of hidden layers.
    activation : str or None or an instance of activation functions defined in
        torch.nn, default = None
        Activation function used in the fully connected output layer. List of supported
        activation functions: ['sigmoid', 'softmax', 'logsoftmax', 'logsigmoid'].
        If None, then no activation function is applied.
    activation_hidden : str or None or an instance of activation functions defined in
        torch.nn, default = "relu"
        The activation function applied inside the hidden layers of the MLP.
        Can be any of "relu", "leakyrelu", "elu", "prelu", "gelu", "selu",
        "rrelu", "celu", "tanh", "hardtanh".
    bias : bool, default = True
        If False, then the layer does not use bias weights.
    dropout : float or tuple of floats, default = (0.1, 0.2, 0.2, 0.3)
        If dropout is a non-zero float, it introduces a Dropout layer on the outputs
        of each hidden layer of the MLP, with dropout probability equal to dropout.
        If dropout is a tuple of floats, it must have length equal to n_layers, and
        each element specifies the dropout probability in the corresponding hidden
        layer of the MLP.
    fc_dropout : float, default = 0.0
        If non-zero, introduces a Dropout layer on the outputs of the fully
        connected output layer of the MLP, with dropout probability equal to fc_dropout.
    random_state   : int, default = 0
        Seed to ensure reproducibility.

    References
    ----------
    .. [1]  Network originally defined in:
    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }
    .. [2] Deep learning for time series classification: a review Fawaz et al. 2019
    .. [3] source code for [2]
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py
    """

    _tags = {
        "authors": ["RecreationalMath"],
        "maintainers": ["RecreationalMath"],
        "python_dependencies": ["torch"],
        "capability:random_state": True,
        "property:randomness": "stochastic",
    }

    def __init__(
        self,
        input_shape: tuple,
        num_classes: int,
        hidden_dim: int = 500,
        n_layers: int = 4,
        activation: str | None | Callable = None,
        activation_hidden: str | None | Callable = "relu",
        bias: bool = True,
        dropout: float | tuple[float, ...] = (0.1, 0.2, 0.2, 0.3),
        fc_dropout: float = 0.0,
        random_state: int = 0,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.bias = bias
        self.dropout = dropout
        # dropout type checking
        if isinstance(self.dropout, tuple):
            if len(self.dropout) != self.n_layers:
                raise ValueError(
                    "If `dropout` is a tuple, then it must have length equal to "
                    f"`n_layers`. Found length of `dropout` to be  {len(self.dropout)}"
                    f" and `n_layers` to be {self.n_layers}."
                )
        elif not isinstance(self.dropout, float):
            raise TypeError(
                "`dropout` should either be of type float or tuple of floats. "
                f"But found the type to be: {type(self.dropout)}"
            )
        self.fc_dropout = fc_dropout
        self.random_state = random_state
        super().__init__()

        # Checking input dimensions
        if isinstance(self.input_shape, tuple):
            if len(self.input_shape) == 3:
                in_features = self.input_shape[1] * self.input_shape[2]
            else:
                raise ValueError(
                    "If `input_shape` is a tuple, it must be of length 3 and in "
                    "format (n_instances, n_dims, series_length). "
                    f"Found length of {len(self.input_shape)}"
                )
        else:
            raise TypeError(
                "`input_shape` should be of type tuple. "
                f"But found the type to be: {type(self.input_shape)}"
            )

        # defining the model architecture
        layers = []

        # defining the input layer
        nnFlatten = _safe_import("torch.nn.Flatten")
        layers.append(nnFlatten())

        prev_dim = in_features
        # defining the hidden layers
        nnLinear = _safe_import("torch.nn.Linear")
        nnDropout = _safe_import("torch.nn.Dropout")
        for i in range(self.n_layers):
            if isinstance(self.dropout, tuple):
                if self.dropout[i] > 0.0:
                    layers.append(nnDropout(self.dropout[i]))
            elif self.dropout > 0.0:
                layers.append(nnDropout(self.dropout))
            layers.append(
                nnLinear(
                    in_features=prev_dim,
                    out_features=self.hidden_dim,
                    bias=self.bias,
                )
            )
            if self.activation_hidden:
                layers.append(self._instantiate_activation(layer="hidden"))
            prev_dim = self.hidden_dim

        # defining the model
        nnSequential = _safe_import("torch.nn.Sequential")
        self.mlp = nnSequential(*layers)

        # defining the output layer
        if self.fc_dropout:
            self.out_dropout = nnDropout(p=self.fc_dropout)
        self.fc = nnLinear(
            in_features=self.hidden_dim,
            out_features=self.num_classes,
        )
        if self.activation:
            self._activation = self._instantiate_activation(layer="output")

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

        out = self.mlp(X)
        if self.fc_dropout:
            out = self.out_dropout(out)
        out = self.fc(out)
        if self.activation:
            out = self._activation(out)
        return out

    def _instantiate_activation(self, layer):
        """Instantiate the activation function to be applied within the MLP.

        Parameters
        ----------
        layer : str
            The name of the layer for which to instantiate the activation function.
            Can be either 'output' or 'hidden'.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied on the output layer.
        """
        if layer == "hidden":
            if isinstance(self.activation_hidden, NNModule):
                return self.activation_hidden
            elif isinstance(self.activation_hidden, str):
                if self.activation_hidden.lower() == "relu":
                    return _safe_import("torch.nn.ReLU")()
                elif self.activation_hidden.lower() == "leakyrelu":
                    return _safe_import("torch.nn.LeakyReLU")()
                elif self.activation_hidden.lower() == "elu":
                    return _safe_import("torch.nn.ELU")()
                elif self.activation_hidden.lower() == "prelu":
                    return _safe_import("torch.nn.PReLU")()
                elif self.activation_hidden.lower() == "gelu":
                    return _safe_import("torch.nn.GELU")()
                elif self.activation_hidden.lower() == "selu":
                    return _safe_import("torch.nn.SELU")()
                elif self.activation_hidden.lower() == "rrelu":
                    return _safe_import("torch.nn.RReLU")()
                elif self.activation_hidden.lower() == "celu":
                    return _safe_import("torch.nn.CELU")()
                elif self.activation_hidden.lower() == "tanh":
                    return _safe_import("torch.nn.Tanh")()
                elif self.activation_hidden.lower() == "hardtanh":
                    return _safe_import("torch.nn.Hardtanh")()
                else:
                    raise ValueError(
                        "If `activation_hidden` is not None, it must be one of "
                        "'relu', 'leakyrelu', 'elu', 'prelu', 'gelu', 'selu', "
                        "'rrelu', 'celu', 'tanh', 'hardtanh'. "
                        f"But found {self.activation_hidden}"
                    )
            else:
                raise TypeError(
                    "`activation_hidden` should either be of type torch.nn.Module or"
                    f" str. But found the type to be: {type(self.activation_hidden)}"
                )
        elif layer == "output":
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
