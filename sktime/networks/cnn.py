# -*- coding: utf-8 -*-
"""Time Convolutional Neural Network (CNN) (minus the final output layer)."""

__author__ = ["James-Large, Withington, TonyBagnall"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class CNNNetwork(BaseDeepNetwork):
    """Establish the network structure for a CNN.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    kernel_size     : int, default = 7
        specifying the length of the 1D convolution window
    avg_pool_size   : int, default = 3
        size of the average pooling windows
    n_conv_layers   : int, default = 2
        the number of convolutional plus average pooling layers
    filter_sizes    : array of int, shape = (nb_conv_layers)
    activation      : string, default = sigmoid
        keras activation function
    random_state    : int, default = 0
        seed to any needed random actions

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    References
    ----------
    .. [1] Zhao et al. Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics 28(1), 162--169, 2017
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        activation="sigmoid",
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")
        self.random_state = random_state
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = [6, 12]
        self.activation = activation

        super(CNNNetwork, self).__init__()

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        # not sure of the whole padding thing
        from tensorflow import keras

        padding = "valid"
        input_layer = keras.layers.Input(input_shape)
        # sort this out, why hard coded to 60?
        if input_shape[0] < 60:
            padding = "same"

        # this does what?
        if len(self.filter_sizes) > self.n_conv_layers:
            self.filter_sizes = self.filter_sizes[: self.n_conv_layers]
        elif len(self.filter_sizes) < self.n_conv_layers:
            self.filter_sizes = self.filter_sizes + [self.filter_sizes[-1]] * (
                self.n_conv_layers - len(self.filter_sizes)
            )
        conv = keras.layers.Conv1D(
            filters=self.filter_sizes[0],
            kernel_size=self.kernel_size,
            padding=padding,
            activation=self.activation,
        )(input_layer)
        conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(conv)

        for i in range(1, self.n_conv_layers):
            conv = keras.layers.Conv1D(
                filters=self.filter_sizes[i],
                kernel_size=self.kernel_size,
                padding=padding,
                activation=self.activation,
            )(conv)
            conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(conv)

        flatten_layer = keras.layers.Flatten()(conv)

        return input_layer, flatten_layer
