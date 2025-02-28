"""Multi Channel Deep Convolution Neural Network (MCDCNN)."""

__author__ = ["James-Large", "Withington"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class MCDCNNNetwork(BaseDeepNetwork):
    """
    Multi Channel Deep Convolutional Neural Network (MCDCNN).

    Adapted from the implementation of Fawaz et. al:
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mcdcnn.py

    Parameters
    ----------
    kernel_size : int, optional (default=5)
        The size of kernel in Conv1D layer.
    pool_size : int, optional (default=2)
        The size of kernel in (Max) Pool layer.
    filter_sizes : tuple, optional (default=(8, 8))
        The sizes of filter for Conv1D layer corresponding
        to each Conv1D in the block.
    dense_units : int, optional (default=732)
        The number of output units of the final Dense
        layer of this Network. This is NOT the final layer
        but the penultimate layer.
    conv_padding : str or None, optional (default="same")
        The type of padding to be applied to convolutional
        layers.
    pool_padding : str or None, optional (default="same")
        The type of padding to be applied to pooling layers.
    random_state : int, optional (default=0)
        The seed to any random action.
    """

    _tags = {
        "authors": ["hfawaz", "James-Large", "Withington"],
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        kernel_size=5,
        pool_size=2,
        filter_sizes=(8, 8),
        dense_units=732,
        conv_padding="same",
        pool_padding="same",
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")
        super().__init__()

        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.filter_sizes = filter_sizes
        self.dense_units = dense_units
        self.conv_padding = conv_padding
        self.pool_padding = pool_padding
        self.random_state = random_state

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data(without batch) fed into the input layer

        Returns
        -------
        input_layers : List of keras.layers.Input of length input_shape[1]
            The input layers of this Network. Equal to number of variables
            in the independent variable (input_shape[1]).
        output_layer: And instance of Keras.layers.Layer
            The output layer of this Network.
        """
        from tensorflow import keras

        n_t = input_shape[0]  # corresponding to the number of time steps (m)
        n_vars = input_shape[1]  # corresponding to the number of variables (d)

        input_layers, conv2_layers = [], []

        for _ in range(n_vars):
            input_layer = keras.layers.Input((n_t, 1))
            input_layers.append(input_layer)

            conv1 = keras.layers.Conv1D(
                self.filter_sizes[0],
                kernel_size=self.kernel_size,
                activation="relu",
                padding=self.conv_padding,
            )(input_layer)
            conv1 = keras.layers.MaxPooling1D(
                pool_size=self.pool_size,
                padding=self.pool_padding,
            )(conv1)

            conv2 = keras.layers.Conv1D(
                self.filter_sizes[1],
                kernel_size=self.kernel_size,
                activation="relu",
                padding=self.conv_padding,
            )(conv1)
            conv2 = keras.layers.MaxPooling1D(
                pool_size=self.pool_size,
                padding=self.pool_padding,
            )(conv2)
            conv2 = keras.layers.Flatten()(conv2)

            conv2_layers.append(conv2)

        # In univariate cases, legacy tf loaders returns just the
        # layer and not a list of layers with one element,
        # therefore simply use that layer, bypassing concat layer.
        if n_vars == 1:
            output_layer = conv2_layers[0]
        else:
            output_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)

        output_layer = keras.layers.Dense(units=self.dense_units, activation="relu")(
            output_layer
        )

        return input_layers, output_layer

    def _prepare_input(self, X):
        # helper function to change X to conform to expected format.
        new_X = []
        n_vars = X.shape[2]

        for i in range(n_vars):
            new_X.append(X[:, :, i : i + 1])

        return new_X
