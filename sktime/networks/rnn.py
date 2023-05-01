# -*- coding: utf-8 -*-
"""Time Recurrent Neural Network (RNN) (minus the final output layer)."""

__authors__ = ["JamesLarge", "Withington", "TonyBagnall", "achieveordie"]

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
        input_shape : int or tuple
            The shape of the data fed into the input layer. It should either
            have dimensions of (m, d) or m. In case an int is passed,
            1 is appended for d.

        Returns
        -------
        output : a compiled Keras Model
        """
        from tensorflow import keras

        if isinstance(input_shape, int):
            input_layer = keras.layers.Input((input_shape, 1))
        elif isinstance(input_shape, tuple):
            if len(input_shape) == 2:
                input_layer = keras.layers.Input(input_shape)
            elif len(input_shape) == 1:
                input_layer = keras.layers.Input((*input_shape, 1))
            else:
                raise ValueError(
                    "If `input_shape` is a tuple, it must either be "
                    f"of length 1 or 2. Found length of be {len(input_shape)}"
                )
        else:
            raise TypeError(
                "`input_shape` should either be of type int or tuple. "
                f"But found the type to be: {type(input_shape)}"
            )

        rnn = keras.layers.SimpleRNN(units=self.units, return_sequences=True)(
            input_layer
        )

        dense = keras.layers.Dense(units=1)(rnn)

        return input_layer, dense
