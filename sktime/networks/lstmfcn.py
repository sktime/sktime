# -*- coding: utf-8 -*-
"""LongShort Term Memory Fully Convolutional Network (LSTM-FCN)."""

__author__ = ["jnrusson1", "solen0id"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class LSTMFCNNetwork(BaseDeepNetwork):
    """
    Implementation of LSTMFCNClassifier from Karim et al (2019) [1].

    Overview
    --------
    Combines an LSTM arm with a CNN arm. Optionally uses an attention mechanism in the
    LSTM which the author indicates provides improved performance.

    Notes
    -----
    Ported from sktime-dl source code
    https://github.com/sktime/sktime-dl/blob/master/sktime_dl/networks/_lstmfcn.py

    References
    ----------
    .. [1] Karim et al. Multivariate LSTM-FCNs for Time Series Classification, 2019
    https://arxiv.org/pdf/1801.04503.pdf
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        kernel_sizes=(8, 5, 3),
        filter_sizes=(128, 256, 128),
        random_state=0,
        lstm_size=8,
        dropout=0.8,
        attention=False,
    ):
        """
        Initialize a new LSTMFCNNetwork object.

        Parameters
        ----------
        kernel_sizes: List[int], default=[8, 5, 3]
            specifying the length of the 1D convolution
         windows
        filter_sizes: List[int], default=[128, 256, 128]
            size of filter for each conv layer
        random_state: int, default=0
            seed to any needed random actions
        lstm_size: int, default=8
            output dimension for LSTM layer
        dropout: float, default=0.8
            controls dropout rate of LSTM layer
        attention: boolean, default=False
            If True, uses custom attention LSTM layer
        """
        self.random_state = random_state
        self.kernel_sizes = kernel_sizes
        self.filter_sizes = filter_sizes
        self.lstm_size = lstm_size
        self.dropout = dropout
        self.attention = attention

        super(LSTMFCNNetwork, self).__init__()

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layers : keras layers
        output_layer : a keras layer
        """
        from tensorflow import keras

        from sktime.networks.lstmfcn_layers import make_attention_lstm

        input_layer = keras.layers.Input(shape=input_shape)

        x = keras.layers.Permute((2, 1))(input_layer)

        if self.attention:
            AttentionLSTM = make_attention_lstm()
            x = AttentionLSTM(self.lstm_size)(x)
        else:
            x = keras.layers.LSTM(self.lstm_size)(x)

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
