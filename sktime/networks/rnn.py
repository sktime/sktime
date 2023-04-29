# -*- coding: utf-8 -*-
"""Time Recurrent Neural Network (RNN) (minus the final output layer)."""

__author__ = "James Large, Withington, Tony Bagnall"

from sktime.networks.base import BaseDeepNetwork


class RNNNetwork(BaseDeepNetwork):
    """Establish the network structure for a CNN.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    random_state    : int, default = 0
        seed to any needed random actions
    """

    def __init__(self, units=50, random_state=0):
        self.random_state = random_state
        self.units = units
        super(RNNNetwork, self).__init__()

    def build_network(self, input_shape, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m,d)

        Returns
        -------
        output : a compiled Keras Model
        """
        from tensorflow import keras

        input_layer = keras.layers.Input((input_shape, 1))

        rnn = keras.layers.SimpleRNN(
            units=self.units, return_sequences=True, input_shape=(input_shape, 1)
        )(input_layer)
        dense = keras.layers.Dense(units=1)(rnn)

        return input_layer, dense
