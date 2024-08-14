"""Time Convolutional Neural Network (CNN) (minus the final output layer)."""

__author__ = [
    "jnrusson1",
]

import math

import numpy as np

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_dl_dependencies, _check_soft_dependencies


class TapNetNetwork(BaseDeepNetwork):
    """Establish Network structure for TapNet.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    kernel_size     : array of int, default = (8, 5, 3)
        specifying the length of the 1D convolution window
    layers          : array of int, default = (500, 300)
        size of dense layers
    filter_sizes    : array of int, shape = (nb_conv_layers), default = (256, 256, 128)
    random_state    : int, default = 1
        seed to any needed random actions
    rp_params       : array of int, default = (-1, 3)
        parameters for random permutation
    dropout         : float, default = 0.5
        dropout rate, in the range [0, 1)
    dilation        : int, default = 1
        dilation value
    padding         : str, default = 'same'
        type of padding for convolution layers
    use_rp          : bool, default = True
        whether to use random projections
    use_att         : bool, default = True
        whether to use self attention
    use_lstm        : bool, default = True
        whether to use an LSTM layer
    use_cnn         : bool, default = True
        whether to use a CNN layer

    References
    ----------
    .. [1] Zhang et al. Tapnet: Multivariate time series classification with
    attentional prototypical network,
    Proceedings of the AAAI Conference on Artificial Intelligence
    34(4), 6845-6852, 2020
    """

    _tags = {"python_dependencies": ["tensorflow", "keras-self-attention"]}

    def __init__(
        self,
        dropout=0.5,
        filter_sizes=(256, 256, 128),
        kernel_size=(8, 5, 3),
        dilation=1,
        layers=(500, 300),
        use_rp=True,
        rp_params=(-1, 3),
        use_att=True,
        use_lstm=True,
        use_cnn=True,
        random_state=1,
        padding="same",
    ):
        _check_soft_dependencies("keras-self-attention", severity="error")
        _check_dl_dependencies(severity="error")

        super().__init__()

        self.random_state = random_state
        self.kernel_size = kernel_size
        self.layers = layers
        self.rp_params = rp_params
        self.filter_sizes = filter_sizes
        self.use_att = use_att
        self.dilation = dilation
        self.padding = padding

        self.dropout = dropout
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # parameters for random projection
        self.use_rp = use_rp
        self.rp_params = rp_params

    @staticmethod
    def output_conv_size(in_size, kernel_size, strides, padding):
        """Get output size from a convolution layer.

        Parameters
        ----------
        in_size         : int
            Dimension of input image, either height or width
        kernel_size     : int
            Size of the convolutional kernel that is applied
        strides         : int
            Stride step between convolution operations
        padding         : int
            Amount of padding done on input.

        Returns
        -------
        output          : int
            Corresponding output dimension after convolution
        """
        # padding removed for now
        output = int((in_size - kernel_size) / strides) + 1

        return output

    @staticmethod
    def euclidean_dist(x, y):
        """Get l2 distance between two points.

        Parameters
        ----------
        x           : 2D array of shape (N x D)
        y           : 2D array of shape (M x D)

        Returns
        -------
        Euclidean distance x and y
        """
        import tensorflow as tf

        # x: N x D
        # y: M x D
        n = tf.shape(x)[0]
        m = tf.shape(y)[0]
        d = tf.shape(x)[1]
        # assert d == tf.shape(y)[1]
        x = tf.expand_dims(x, 1)
        y = tf.expand_dims(y, 0)
        x = tf.broadcast_to(x, shape=(n, m, d))
        y = tf.broadcast_to(y, shape=(n, m, d))
        return tf.math.reduce_sum(tf.math.pow(x - y, 2), axis=2)

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        --------
        input_shape: tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer  : a keras layer
        output_layer : a keras layer
        """
        import tensorflow as tf
        from keras_self_attention import SeqSelfAttention
        from tensorflow import keras

        input_layer = keras.layers.Input(input_shape)

        if self.rp_params[0] < 0:
            dim = input_shape[0]
            self.rp_params = [3, math.floor(dim * 2 / 3)]
        self.rp_group, self.rp_dim = self.rp_params

        if self.use_lstm:
            self.lstm_dim = 128

            x_lstm = keras.layers.LSTM(self.lstm_dim, return_sequences=True)(
                input_layer
            )
            x_lstm = keras.layers.Dropout(0.8)(x_lstm)

            if self.use_att:
                x_lstm = SeqSelfAttention(128, attention_type="multiplicative")(x_lstm)
                # pass
            x_lstm = keras.layers.GlobalAveragePooling1D()(x_lstm)

        if self.use_cnn:
            # Convolutional Network
            # input ts: # N * C * L
            if self.use_rp:
                self.conv_1_models = keras.Sequential()

                for i in range(self.rp_group):
                    self.idx = np.random.permutation(input_shape[1])[0 : self.rp_dim]
                    channel = keras.layers.Lambda(
                        lambda x: tf.gather(x, indices=self.idx, axis=2)
                    )(input_layer)
                    # x_conv = x
                    # x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                    x_conv = keras.layers.Conv1D(
                        self.filter_sizes[0],
                        kernel_size=self.kernel_size[0],
                        dilation_rate=self.dilation,
                        strides=1,
                        padding=self.padding,
                    )(channel)  # N * C * L

                    x_conv = keras.layers.BatchNormalization()(x_conv)
                    x_conv = keras.layers.LeakyReLU()(x_conv)

                    x_conv = keras.layers.Conv1D(
                        self.filter_sizes[1],
                        kernel_size=self.kernel_size[0],
                        dilation_rate=self.dilation,
                        strides=1,
                        padding=self.padding,
                    )(x_conv)
                    x_conv = keras.layers.BatchNormalization()(x_conv)
                    x_conv = keras.layers.LeakyReLU()(x_conv)

                    x_conv = keras.layers.Conv1D(
                        self.filter_sizes[2],
                        kernel_size=self.kernel_size[0],
                        dilation_rate=self.dilation,
                        strides=1,
                        padding=self.padding,
                    )(x_conv)
                    x_conv = keras.layers.BatchNormalization()(x_conv)
                    x_conv = keras.layers.LeakyReLU()(x_conv)
                    if self.use_att:
                        x_conv = SeqSelfAttention(128, attention_type="multiplicative")(
                            x_conv
                        )
                        # pass

                    x_conv = keras.layers.GlobalAveragePooling1D()(x_conv)

                    if i == 0:
                        x_conv_sum = x_conv
                    else:
                        x_conv_sum = keras.layers.Concatenate()([x_conv_sum, x_conv])

                x_conv = x_conv_sum

            else:
                x_conv = keras.layers.Conv1D(
                    self.filter_sizes[0],
                    kernel_size=self.kernel_size[0],
                    dilation_rate=self.dilation,
                    strides=1,
                    padding=self.padding,
                )(input_layer)  # N * C * L

                x_conv = keras.layers.BatchNormalization()(x_conv)
                x_conv = keras.layers.LeakyReLU()(x_conv)

                x_conv = keras.layers.Conv1D(
                    self.filter_sizes[1],
                    kernel_size=self.kernel_size[0],
                    dilation_rate=self.dilation,
                    strides=1,
                    padding=self.padding,
                )(x_conv)
                x_conv = keras.layers.BatchNormalization()(x_conv)
                x_conv = keras.layers.LeakyReLU()(x_conv)

                x_conv = keras.layers.Conv1D(
                    self.filter_sizes[2],
                    kernel_size=self.kernel_size[0],
                    dilation_rate=self.dilation,
                    strides=1,
                    padding=self.padding,
                )(x_conv)
                x_conv = keras.layers.BatchNormalization()(x_conv)
                x_conv = keras.layers.LeakyReLU()(x_conv)
                if self.use_att:
                    x_conv = SeqSelfAttention(128)(x_conv)
                    # pass

                x_conv = keras.layers.GlobalAveragePooling1D()(x_conv)

        if self.use_lstm and self.use_cnn:
            x = keras.layers.Concatenate()([x_conv, x_lstm])
        elif self.use_lstm:
            x = x_lstm
        elif self.use_cnn:
            x = x_conv

        # Mapping section
        x = keras.layers.Dense(self.layers[0], name="fc_")(x)
        x = keras.layers.LeakyReLU(name="relu_")(x)
        x = keras.layers.BatchNormalization(name="bn_")(x)

        x = keras.layers.Dense(self.layers[1], name="fc_2")(x)

        return input_layer, x
