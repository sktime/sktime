"""Multi Layer Perceptron (MLP) (minus the final output layer) in TensorFlow."""

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class MLPNetwork(BaseDeepNetwork):
    """Establish the network structure for a MLP.

    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

    Implements a simple MLP network, as in [1]_.

    Parameters
    ----------
    random_state : int, default = 0
        seed to any needed random actions
    activation : string, default = "relu"
        activation function used for hidden layers;
        List of available keras activation functions:
        https://keras.io/api/layers/activations/
    dropout : float or tuple, default=(0.1, 0.2, 0.2, 0.3)
        The dropout rate for the hidden layers.
        If float, the same rate is used for all layers.
        If tuple, length must equal n_layers + 1, where the first
        n_layers elements correspond to dropout before each hidden
        Dense layer, and the last element is the trailing dropout
        after the final hidden layer.
    n_layers : int, default=3
        Number of hidden Dense layers in the MLP.
    hidden_dim : int, default=500
        Number of units in each hidden Dense layer.

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
        "authors": ["hfawaz", "James-Large", "Withington", "AurumnPegasus", "noxthot"],
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        random_state=0,
        activation="relu",
        dropout=(0.1, 0.2, 0.2, 0.3),
        n_layers=3,
        hidden_dim=500,
    ):
        _check_dl_dependencies(severity="error")
        self.activation = activation
        self.random_state = random_state
        self.dropout = dropout
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        super().__init__()

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple of shape = (series_length (m), n_dimensions (d))
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        if isinstance(self.dropout, float):
            _dropout = (self.dropout,) * (self.n_layers + 1)
        elif isinstance(self.dropout, tuple):
            if len(self.dropout) != self.n_layers + 1:
                raise ValueError(
                    "If `dropout` is a tuple, it must have length equal to the "
                    "number of hidden layers in the MLP, where each element "
                    "specifies the rate for the corresponding layer. "
                    f"tuple must be of length n_layers + 1 = {self.n_layers + 1}. "
                    f"Found length of dropout to be: {len(self.dropout)}."
                )
            _dropout = self.dropout
        else:
            raise TypeError(
                "`dropout` should either be of type float or tuple. "
                f"But found the type to be: {type(self.dropout)}"
            )

        from tensorflow import keras

        # flattened because multivariate should be on same axis
        input_layer = keras.layers.Input(input_shape)
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        x = input_layer_flattened
        for i in range(self.n_layers):
            x = keras.layers.Dropout(_dropout[i])(x)
            x = keras.layers.Dense(self.hidden_dim, activation=self.activation)(x)

        output_layer = keras.layers.Dropout(_dropout[self.n_layers])(x)
        return input_layer, output_layer
