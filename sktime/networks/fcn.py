# -*- coding: utf-8 -*-
"""Fully Connected Neural Network (FCN) (minus the final output layer)."""

__author__ = ["James-Large", "AurumnPegasus"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class FCNNetwork(BaseDeepNetwork):
    """Establish the network structure for a FCN.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    random_state    : int, default = 0
        seed to any needed random actions

    Notes
    -----
    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py

    References
    ----------
    .. [1] Network originally defined in:
    @inproceedings{wang2017time,
      title={Time series classification from scratch with deep neural networks:
       A strong baseline},
      author={Wang, Zhiguang and Yan, Weizhong and Oates, Tim},
      booktitle={2017 International joint conference on neural networks
      (IJCNN)},
      pages={1578--1585},
      year={2017},
      organization={IEEE}
    }
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        random_state=0,
    ):
        super(FCNNetwork, self).__init__()
        _check_dl_dependencies(severity="error")
        self.random_state = random_state

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
        from tensorflow import keras

        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding="same")(
            input_layer
        )
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation="relu")(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation(activation="relu")(conv2)

        conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation(activation="relu")(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        return input_layer, gap_layer
