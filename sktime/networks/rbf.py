"""RBF Neural Networks for Time Series Forecasting."""

from sktime.utils.dependencies._dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn

else:

    class nn:
        """dummy class if torch is not available."""

        class Module:
            """dummy class if torch is not available."""

            def __init__(self, *args, **kwargs):
                raise ImportError("torch is not available. Please install torch first.")


class RBFLayer(nn.Module):
    r"""RBF layer to transform input data into a new feature space.

    This layer applies an RBF transformation to each input feature,
    expanding the feature space based on distances to predefined center points.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features (number of RBF centers)
    centers : torch.Tensor, optional (default=None)
        The centers :math:`c_k` for the RBF transformation.
        If None, centers are evenly spaced.
    gamma : float, optional (default=1.0)
        Parameter controlling the spread of the RBFs
    rbf_type : str, optional (default="gaussian")
        The type of RBF kernel to apply.

        - "gaussian": :math:`\exp(-\gamma (t - c)^2)`
        - "multiquadric": :math:`\sqrt{1 + \gamma (t - c)^2}`
        - "inverse_multiquadric": :math:`1 / \sqrt{1 + \gamma (t - c)^2}`

    """

    def __init__(
        self, in_features, out_features, centers=None, gamma=1.0, rbf_type="gaussian"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma
        self.rbf_type = rbf_type.lower()

        if centers is None:
            centers = torch.linspace(-1, 1, out_features).reshape(-1, 1)
            centers = centers.repeat(1, in_features)
        else:
            centers = torch.as_tensor(centers, dtype=torch.float32)

        self.centers = nn.Parameter(centers, requires_grad=True)

        valid_rbf_types = {"gaussian", "multiquadric", "inverse_multiquadric"}
        if self.rbf_type not in valid_rbf_types:
            raise ValueError(
                f"rbf_type must be one of {valid_rbf_types}, got {self.rbf_type}"
            )

    def forward(self, x):
        """Apply the RBF transformation to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor with RBF features.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)

        distances_squared = torch.sum(diff**2, dim=-1)

        if self.rbf_type == "gaussian":
            return torch.exp(-self.gamma * distances_squared)
        elif self.rbf_type == "multiquadric":
            return torch.sqrt(1 + self.gamma * distances_squared)
        else:  # inverse_multiquadric
            return 1 / torch.sqrt(1 + self.gamma * distances_squared)


class RBFNetwork(nn.Module):
    r"""Neural network with an RBF layer followed by fully connected layers.

    This model is designed to use RBF-transformed features as input for a series
    of linear transformations, enabling effective learning from non-linear
    representations.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of units in the RBF layer.
    output_size : int
        Number of output features for the network.
    centers : torch.Tensor, optional (default=None)
        Centers points for the RBF layer
    gamma : float, optional (default=1.0)
        Scaling factor controlling the spread of the RBF layer.
    rbf_type : str, optional (default="gaussian")
        The type of RBF kernel to apply.

        - "gaussian": :math:`\exp(-\gamma (t - c)^2)`
        - "multiquadric": :math:`\sqrt{1 + \gamma (t - c)^2}`
        - "inverse_multiquadric": :math:`1 / \sqrt{1 + \gamma (t - c)^2}`

    hidden_layers : list of int, optional (default=[64, 32])
        Sizes of linear layers following the RBF layer
    mode : {"ar", "direct"}, optional (default="ar")
        Mode of operation for the network:

        - "ar": Outputs a single value for autoregressive forecasting.
        - "direct": Outputs multiple values for direct forecasting.

    activation : str, optional (default="relu")
        Activation function to apply after each linear layer. Supported values are:
        "relu", "leaky_relu", "elu", "selu", "tanh", "sigmoid", "gelu".
    dropout_rate : float, optional (default=0.1)
        Dropout rate applied after each hidden layer. A value of 0 disables dropout.

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        centers=None,
        gamma=1.0,
        rbf_type="gaussian",
        hidden_layers=[64, 32],
        mode="ar",
        activation="relu",
        dropout_rate=0.1,
    ):
        super().__init__()
        self.mode = mode

        self.rbf_layer = RBFLayer(
            in_features=input_size,
            out_features=hidden_size,
            centers=centers,
            gamma=gamma,
            rbf_type=rbf_type,
        )

        activation_fn = self._get_activation_fn(activation)

        layers = []
        prev_size = hidden_size

        for size in hidden_layers:
            layer_group = [nn.Linear(prev_size, size), activation_fn]

            if dropout_rate > 0:
                layer_group.append(nn.Dropout(dropout_rate))

            layers.extend(layer_group)
            prev_size = size

        if mode == "direct":
            layers.append(nn.Linear(prev_size, output_size))
        else:  # "ar" mode
            layers.append(nn.Linear(prev_size, 1))

        self.sequential_layers = nn.Sequential(*layers)

    def _get_activation_fn(self, activation_name):
        """Get activation function by name.

        Parameters
        ----------
        activation_name : str
            Name of the activation function.

        Returns
        -------
        nn.Module
            PyTorch activation function module.

        """
        activation_fns = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
        }
        activation_name = activation_name.lower()

        if activation_name not in activation_fns:
            raise ValueError(
                f"Unsupported activation function: {activation_name}. "
                f"Supported functions are: {list(activation_fns.keys())}"
            )

        return activation_fns[activation_name]

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        x = self.rbf_layer(x)
        return self.sequential_layers(x)
