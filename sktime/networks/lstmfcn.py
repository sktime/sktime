# -*- coding: utf-8 -*-
"""LSTM-FCN Network (minus the final output layer)."""

__author__ = ["Jack Russon", "AurumnPegasus"]

from sktime.networks.base import BaseDeepNetwork
from sktime.networks.modules.attentionlstm import AttentionLSTM
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class LSTMFCNNetwork(BaseDeepNetwork):
    """Establish the network structure for a LSTM-FCN.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    kernel_sizes     : tuple, default = (8, 5, 3)
        specifying the length of the 1D convolution window
    filter_sizes   : tuple, default = (128, 256, 128)
        size of filter for each conv layer
    num_cell        : int, default = 8,
        number of cells in LSTM layer
    dropout      : float, default = 0.8
        dropout for the network
    random_state    : int, default = 0
        seed to any needed random actions

    Notes
    -----
    Implementation of LSTMFCNClassifier from Karim et al (2019).

    References
    ----------
    .. [1] @article{Karim_2019,
    title={Multivariate LSTM-FCNs for time series classification},
    volume={116},
    ISSN={0893-6080},
    url={http://dx.doi.org/10.1016/j.neunet.2019.04.014},
    DOI={10.1016/j.neunet.2019.04.014},
    journal={Neural Networks},
    publisher={Elsevier BV},
    author={Karim, Fazle and Majumdar, Somshubra and Darabi,
     Houshang and Harford, Samuel},
    year={2019},
    month={Aug},
    pages={237–245}
    }
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        kernel_sizes=(8, 5, 3),
        filter_sizes=(128, 256, 128),
        num_cells=8,
        dropout=0.8,
        random_state=0,
        attention=False,
    ):
        _check_dl_dependencies(severity="error")
        self.random_state = random_state
        self.kernel_sizes = kernel_sizes
        self.filter_sizes = filter_sizes
        self.num_cells = num_cells
        self.dropout = dropout
        self.random_state = random_state
        self.attention = attention

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
        from tensorflow import keras

        input_layer = keras.layers.Input(shape=input_shape)

        x = keras.layers.Permute((2, 1))(input_layer)
        if self.attention:
            x = AttentionLSTM(self.num_cells)(x)
        else:
            x = keras.layers.LSTM(self.num_cells)(x)
        x = keras.layers.Dropout(self.dropout)(x)

        y = keras.layers.Conv1D(
            self.filter_sizes[0],
            self.kernel_sizes[0],
            padding="same",
            kernel_initializer="he_uniform",
        )(input_layer)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)

        y = keras.layers.Conv1D(
            self.filter_sizes[1],
            self.kernel_sizes[1],
            padding="same",
            kernel_initializer="he_uniform",
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)

        y = keras.layers.Conv1D(
            self.filter_sizes[2],
            self.kernel_sizes[2],
            padding="same",
            kernel_initializer="he_uniform",
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)

        y = keras.layers.GlobalAveragePooling1D()(y)

        output_layer = keras.layers.concatenate([x, y])
        return input_layer, output_layer
