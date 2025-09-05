"""Multi-layer Perceptron (MLP) Network in PyTorch for classification & regression."""

__author__ = ["RecreationalMath"]

from sktime.networks.base_torch import BasePytorchDeepNetwork
from sktime.utils.dependencies import _check_torch_dependencies

if _check_torch_dependencies("torch", severity="none"):
    import torch.nn as nn


class PyTorchMLPNetwork(nn.Module, BasePytorchDeepNetwork):
    """
    Establish the network structure for a MLP Network in PyTorch.

    Parameters
    ----------
    should inherited fields be listed here?
    input_size      : int
        Number of input features
    output_size    : int
        Number of output classes
    hidden_size     : int
        Number of hidden units in the hidden layer
    n_hidden_layers : int
        Number of hidden layers
    activation_hidden      : string or a torch callable
        Activation function to be used in the hidden layer
    activation_output      : string or a torch callable
        Whether you explicitly apply an activation function
        on the output layer in PyTorch or not,
        depends on the specific loss function you are using
        and the nature of task (e.g., classification, regression)
        Check PyTorch documentation for more details:
        https://docs.pytorch.org/docs/stable/nn.html#loss-functions
    dropout        : float, default = 0.0
        Dropout rate for regularization
    use_bias        : boolean, default = True
        whether the layer uses a bias vector
    random_state    : int or None, default=None
        Seed for reproducibility
    """

    _tags = {
        "authors": ["RecreationalMath"],
        "python_dependencies": "torch",
    }

    # orig params: self, random_state=0
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 500,
        n_hidden_layers: int = 3,
        activation_hidden="relu",
        activation_output="relu",
        dropout=0.0,
        use_bias=True,
        random_state=0,
    ):
        # checking if PyTorch dependencies are missing
        _check_torch_dependencies(severity="error")

        # initializing the base class
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.dropout = dropout
        self.use_bias = use_bias
        self.random_state = random_state

        # defining the model architecture
        layers = []

        # defining the input layer
        input_layer = nn.Flatten()
        layers.append(input_layer)

        prev_dim = self.input_size

        # defining the hidden layers
        for _ in range(self.n_hidden_layers):
            layers.append(
                nn.Linear(
                    in_features=prev_dim,
                    out_features=self.hidden_size,
                    bias=self.use_bias,
                )
            )
            layers.append(self._get_activation(self.activation_hidden))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = self.hidden_size

        # defining the output layer
        layers.append(
            nn.Linear(
                in_features=prev_dim, out_features=self.output_size, bias=self.use_bias
            )
        )
        layers.append(self._get_activation(self.activation_output))

        # defining the model
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        """
        Perform a forward pass through the MLP network.

        Parameters
        ----------
        input : tensor
            The input data fed into the input layer

        Returns
        -------
        output : tensor
            The output of the MLP network
        """
        return self.model(X)

    def _get_activation(self, activation):
        """Get the callable object of the activation function.

        Parameters
        ----------
        input : str or callable object
            The activation function name or a callable object

        Returns
        -------
        output : callable object
            The callable object of the activation function
        """
        if isinstance(activation, str):
            if activation.lower() == "relu":
                return nn.ReLU()
            elif activation.lower() == "sigmoid":
                return nn.Sigmoid()
            elif activation.lower() == "tanh":
                return nn.Tanh()
            elif activation.lower() == "leakyrelu":
                return nn.LeakyReLU()
            elif activation.lower() == "elu":
                return nn.ELU()
            elif activation.lower() == "prelu":
                return nn.PReLU()
            elif activation.lower() == "softmax":
                return nn.Softmax(dim=1)
            else:
                raise ValueError(f"Unknown activation function: {activation}")
        elif callable(activation):
            return activation
        else:
            raise TypeError(f"Invalid activation type: {type(activation)}")
