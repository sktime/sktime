"""Multi Layer Perceptron (MLP) in PyTorch."""

__authors__ = ["Faakhir30"]
__all__ = ["MLPNetworkTorch"]

from collections.abc import Callable
from typing import Literal

import numpy as np

from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class MLPNetworkTorch(NNModule):
    """Establish the network structure for an MLP in PyTorch.

    Implements a simple MLP network, as in [1]_.

    Parameters
    ----------
    input_size : int or tuple
        Number of expected features in the input.
        If int, it represents the flattened input size.
        If tuple, it should be of shape (series_length, n_dimensions).
    num_classes : int
        Number of classes to predict
    task : Literal["classification", "regression"], default = "classification"
        The task type of the network. Networks output shape depends on the task type.
    use_bias : bool, default = True
        Whether to use bias in the fully connected layers.
    activation : str or None or an instance of activation functions defined in
        torch.nn, default = None
        Activation function used in the fully connected output layer.
    activation_hidden : str, default = "relu"
        Activation function used for hidden layers.
        List of available PyTorch activation functions:
        https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
    dropout : float or tuple, default=(0.1, 0.2, 0.2, 0.3)
        The dropout rate for the hidden layers.
        If float, the same rate is used for all layers.
        If tuple, it must have length equal to number of hidden layers in the MLP,
        each element specifying the dropout rate for the corresponding hidden layer.
        Current implementation of the MLP has 4 hidden layers.
    random_state : int, default = 0
        Seed to ensure reproducibility.

    References
    ----------
    .. [1]  Network originally defined in:
    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }
    """

    _tags = {
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
        task: Literal["classification", "regression"] = "classification",
        activation: str | None | Callable = None,
        activation_hidden: str = "relu",
        use_bias: bool = True,
        dropout: float | tuple = (0.1, 0.2, 0.2, 0.3),
        random_state: int = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.task = task
        self.random_state = random_state
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.use_bias = use_bias
        self.dropout = dropout

        # Validate and process dropout
        if isinstance(self.dropout, float):
            _dropout = (self.dropout,) * 4
        elif isinstance(self.dropout, tuple):
            if len(self.dropout) != 4:
                raise ValueError(
                    "If `dropout` is a tuple, it must have length equal to the "
                    "number of hidden layers in the MLP, where each element "
                    "specifies the rate for the corresponding layer. "
                    "The current implementation of the MLP has 4 hidden layers, "
                    f"hence the tuple must be of length 4. "
                    f"Found length of dropout to be: {len(self.dropout)}."
                )
            _dropout = self.dropout
        else:
            raise TypeError(
                "`dropout` should either be of type float or tuple. "
                f"But found the type to be: {type(self.dropout)}"
            )

        # Calculate flattened input size
        if isinstance(self.input_size, int):
            in_features = self.input_size
        elif isinstance(self.input_size, tuple):
            # Flatten: (series_length, n_dimensions) -> series_length * n_dimensions
            in_features = np.prod(self.input_size)
        else:
            raise TypeError(
                "`input_size` should either be of type int or tuple. "
                f"But found the type to be: {type(self.input_size)}"
            )

        # Instantiate activation functions for hidden layers and output layer
        self._activation_hidden_fn = self._instantiate_activation(
            self.activation_hidden
        )
        if self.activation:
            self._activation_fn = self._instantiate_activation(self.activation)

        # Build the network layers
        nnLinear = _safe_import("torch.nn.Linear")
        nnDropout = _safe_import("torch.nn.Dropout")

        # First layer: dropout -> dense
        self.dropout1 = nnDropout(p=_dropout[0])
        self.layer1 = nnLinear(in_features, 500)

        # Second layer: dropout -> dense
        self.dropout2 = nnDropout(p=_dropout[1])
        self.layer2 = nnLinear(500, 500)

        # Third layer: dropout -> dense
        self.dropout3 = nnDropout(p=_dropout[2])
        self.layer3 = nnLinear(500, 500)

        # Fourth layer: dropout -> output layer
        self.dropout4 = nnDropout(p=_dropout[3])
        self.fc = nnLinear(500, num_classes, bias=use_bias)

        # Set random seed for reproducibility
        if self.random_state is not None:
            torchManual_seed = _safe_import("torch.manual_seed")
            torchManual_seed(self.random_state)

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor. If input_size is a tuple, X should be of shape
            (batch_size, series_length, n_dimensions).
            If input_size is an int, X should be of shape (batch_size, input_size).

        Returns
        -------
        out : torch.Tensor
            Output tensor after passing through the MLP layers and final output layer.
            Shape: (batch_size, num_classes) for classification and (batch_size,)
            for regression if num_classes == 1.
        """
        if isinstance(X, np.ndarray):
            torchFrom_numpy = _safe_import("torch.from_numpy")
            X = torchFrom_numpy(X).float()

        # Flatten input if needed (for multivariate time series)
        if isinstance(self.input_size, tuple):
            # X shape: (batch_size, series_length, n_dimensions)
            batch_size = X.size(0)
            X = X.view(
                batch_size, -1
            )  # Flatten to (batch_size, series_length * n_dimensions)

        # First layer
        out = self.dropout1(X)
        out = self.layer1(out)
        out = self._activation_hidden_fn(out)

        # Second layer
        out = self.dropout2(out)
        out = self.layer2(out)
        out = self._activation_hidden_fn(out)

        # Third layer
        out = self.dropout3(out)
        out = self.layer3(out)
        out = self._activation_hidden_fn(out)

        # Fourth dropout -> final output layer
        out = self.dropout4(out)
        out = self.fc(out)

        if self.activation:
            out = self._activation_fn(out)

        # For regression (num_classes == 1),
        # squeeze to (batch_size,) to match target shape
        if self.task == "regression" and self.num_classes == 1:
            out = out.squeeze(-1)

        return out

    def _instantiate_activation(self, activation: str | None):
        """Instantiate the activation function to be applied in hidden layers.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied in hidden layers.
        """
        if isinstance(activation, NNModule):
            return activation
        elif isinstance(activation, str):
            activation_lower = activation.lower()
            if activation_lower == "relu":
                return _safe_import("torch.nn.ReLU")()
            elif activation_lower == "tanh":
                return _safe_import("torch.nn.Tanh")()
            elif activation_lower == "sigmoid":
                return _safe_import("torch.nn.Sigmoid")()
            elif activation_lower == "gelu":
                return _safe_import("torch.nn.GELU")()
            elif activation_lower == "elu":
                return _safe_import("torch.nn.ELU")()
            elif activation_lower == "leakyrelu":
                return _safe_import("torch.nn.LeakyReLU")()
            else:
                raise ValueError(
                    f"Unsupported activation function: {activation}. "
                    "Supported activations:"
                    "'relu', 'tanh', 'sigmoid', 'gelu', 'elu', 'leakyrelu'"
                )
        else:
            raise TypeError(
                "`activation` should either be of type str or torch.nn.Module. "
                f"But found the type to be: {type(activation)}"
            )
