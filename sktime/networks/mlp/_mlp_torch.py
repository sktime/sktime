"""MLP (Multi-Layer Perceptron) deep learning network structure in PyTorch.

For Classification and Regression.
"""

__authors__ = ["RecreationalMath"]
__all__ = ["MLPNetworkTorch"]

from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")

class MLPNetworkTorch(NNModule):
    """Establish the network structure for an MLP in PyTorch.

    Parameters
    ----------
    input_size : int
        Number of expected features in the input
    num_classes : int
        Number of classes to predict
    hidden_dim : int, default = 500
        Number of features in the hidden state
    n_layers : int, default = 3
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
        "authors": ["RecreationalMath"],
        "maintainers": ["RecreationalMath"],
        "python_version": ">=3.10",
        "python_dependencies": ["torch"],
        "capability:random_state": True,
        "property:randomness": "stochastic",
    }

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_dim: int = 500,
        n_layers: int = 3,
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
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
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
                    "If `input_size` is a tuple, it must either be of length 3 and in "
                    "format (n_instances, n_dims, series_length). "
                    f"Found length of {len(self.input_size)}"
                )
        else:
            raise TypeError(
                "`input_size` should either be of type int or tuple. "
                f"But found the type to be: {type(self.input_size)}"
            )

        # defining the model architecture
        layers = []

        # defining the input layer
        nnFlatten = _safe_import("torch.nn.Flatten")
        input_layer = nnFlatten()
        layers.append(input_layer)

        prev_dim = in_features

        # defining the hidden layers
        nnLinear = _safe_import("torch.nn.Linear")
        nnDropout = _safe_import("torch.nn.Dropout")
        for _ in range(self.n_layers):
            layers.append(
                nnLinear(
                    in_features=prev_dim,
                    out_features=self.hidden_dim,
                    bias=self.bias,
                )
            )
            if self.activation_hidden:
                layers.append(self._instantiate_activation())
            if self.dropout > 0.0:
                layers.append(nnDropout(self.dropout))
            prev_dim = self.hidden_dim

        # defining the model
        nnSequential = _safe_import("torch.nn.Sequential")
        self.mlp = nnSequential(*layers)

        # defining the output layer
        self.fc_dropout = fc_dropout
        self.num_classes = num_classes
        self.init_weights = init_weights
        if self.fc_dropout:
            self.out_dropout = nnDropout(p=self.fc_dropout)
        self.fc = nnLinear(
            in_features=self.hidden_dim * (2 if self.bidirectional else 1),
            out_features=self.num_classes,
        )
        if self.activation:
            self._activation = self._instantiate_activation()
        if self.init_weights:
            self.apply(self._init_weights)