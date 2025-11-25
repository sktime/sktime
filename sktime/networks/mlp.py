"""Multi Layer Perceptron (MLP) (minus the final output layer)."""

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
        If tuple, it must have length equal to number of hidden layers in the MLP,
        each element specifying the dropout rate for the corresponding hidden layer.
        Current implementation of the MLP has 4 hidden layers.

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

    def __init__(self, random_state=0, activation="relu", dropout=(0.1, 0.2, 0.2, 0.3)):
        _check_dl_dependencies(severity="error")
        self.activation = activation
        self.random_state = random_state
        self.dropout = dropout
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
        if isinstance(self.dropout, (float, int)):
            dropout_rates = [float(self.dropout)] * 4
        elif isinstance(self.dropout, tuple):
            if len(self.dropout) != 4:
                raise ValueError(
                    "If `dropout` is a tuple, it must be of length 4"
                    "for MLPNetwork. "
                    f"Found length of {len(self.dropout)}"
                )
            dropout_rates = self.dropout
        else:
            raise TypeError(
                "`dropout` should either be of type float or tuple. "
                f"But found the type to be: {type(self.dropout)}"
            )

        from tensorflow import keras

        # flattened because multivariate should be on same axis
        input_layer = keras.layers.Input(input_shape)
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(dropout_rates[0])(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation=self.activation)(layer_1)

        layer_2 = keras.layers.Dropout(dropout_rates[1])(layer_1)
        layer_2 = keras.layers.Dense(500, activation=self.activation)(layer_2)

        layer_3 = keras.layers.Dropout(dropout_rates[2])(layer_2)
        layer_3 = keras.layers.Dense(500, activation=self.activation)(layer_3)

        output_layer = keras.layers.Dropout(dropout_rates[3])(layer_3)

        return input_layer, output_layer
