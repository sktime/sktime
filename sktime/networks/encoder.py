# -*- coding: utf-8 -*-
"""Encoder Network (minus the final output layer)."""

__author__ = ["James-Large", "Withington", "AurumnPegasus"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import (
    _check_dl_dependencies,
    _check_soft_dependencies,
)

_check_dl_dependencies(severity="warning")
_check_soft_dependencies(severity="warning")


class EncoderNetwork(BaseDeepNetwork):
    """Establish the network structure for a Encoder.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    random_state    : int, default = 0
        seed to any needed random actions

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/encoder.py

    References
    ----------
    .. [1]  Network originally defined in:
    @article{serra2018towards,
       title={Towards a universal neural network encoder for time series},
       author={Serrà, J and Pascual, S and Karatzoglou, A},
       journal={Artif Intell Res Dev Curr Chall New Trends Appl},
       volume={308},
       pages={120},
       year={2018}
    }
    """

    _tags = {"python_dependencies": ["tensorflow", "tensorflow_addons"]}

    def __init__(
        self,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")
        self.random_state = random_state
        super(EncoderNetwork, self).__init__()

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
        import tensorflow_addons as ADDONS
        from tensorflow import keras

        input_layer = keras.layers.Input(input_shape)

        # First Convolution Block
        conv1 = keras.layers.Conv1D(
            filters=128, kernel_size=5, strides=1, padding="same"
        )(input_layer)
        conv1 = ADDONS.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2, padding="same")(conv1)

        # Second Convolution Block
        conv2 = keras.layers.Conv1D(
            filters=256, kernel_size=11, strides=1, padding="same"
        )(conv1)
        conv2 = ADDONS.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2, padding="same")(conv2)

        # Third Convolution Block
        conv3 = keras.layers.Conv1D(
            filters=512, kernel_size=21, strides=1, padding="same"
        )(conv2)
        conv3 = ADDONS.layers.InstanceNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)

        # Split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(conv3)

        # Attention Mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])

        # Final Layer
        dense_layer = keras.layers.Dense(units=256, activation="sigmoid")(
            multiply_layer
        )
        dense_layer = ADDONS.layers.InstanceNormalization()(dense_layer)

        # Output Layer
        flatten_layer = keras.layers.Flatten()(dense_layer)

        return input_layer, flatten_layer
