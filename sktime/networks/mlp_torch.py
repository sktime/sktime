"""Multi Layer Perceptron (MLP) Network for Classification and Regression."""

__author__ = ["Jack Russon"]

from sktime.utils.dependencies import _check_dl_dependencies

if _check_dl_dependencies("torch", severity="none"):
    import torch.nn as nn


class PyTorchMLPNetwork(nn.Module):
    """Multi Layer Perceptron Network for Classification and Regression."""

    def __init__(
        self,
        input_shape,
        num_classes=2,
        hidden_dims=None,
        activation="relu",
        dropout=0.0,
        use_bias=True,
    ):
        super().__init__()

        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [500, 500, 500]

        in_features = (
            input_shape[0] * input_shape[1] if len(input_shape) > 1 else input_shape[0]
        )

        # Build the base network (hidden layers)
        layers = []
        layers.append(nn.Flatten())  # Start with flattening
        prev_dim = in_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=use_bias))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.base_network = nn.Sequential(*layers)

        # Add the output layer
        # For classification: num_classes outputs
        # For regression: 1 output (continuous value)
        self.output_layer = nn.Linear(prev_dim, num_classes, bias=use_bias)

        # Store whether this is for regression (1 output) or classification (multiple outputs)
        self.is_regression = num_classes == 1

    def _get_activation(self, activation):
        """Get the activation function based on the string name."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU()
        elif activation.lower() == "elu":
            return nn.ELU()
        else:
            raise ValueError(
                f"Unsupported activation function: {activation}. "
                f"Supported: relu, tanh, sigmoid, leaky_relu, elu"
            )

    def forward(self, X):
        """Forward pass through the network."""
        x = self.base_network(X)
        x = self.output_layer(x)
        return x
