"""Time Convolutional Neural Network (CNN) (minus the final output layer)."""

import math
import warnings

import numpy as np

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class TapNetNetwork(BaseDeepNetwork):
    """Establish Network structure for TapNet.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    activation_hidden : str or callable, default = "leaky_relu"
        activation function to use in the hidden layers;
        List of available keras activation functions:
        https://keras.io/api/layers/activations/
    kernel_size : tuple of int, default = (8, 5, 3)
        specifying the length of the 1D convolution window
    layers : tuple of int, default = (500, 300)
        sizes of dense layers
    filter_sizes : tuple of int, default = (256, 256, 128)
        number of convolutional filters in each conv block
    random_state : int or None, default = None
        seed to any needed random actions
    rp_group : int, default = 3
        number of random permutation groups g for random dimension permutation
    rp_alpha : float, default = 2.0
        scale factor alpha used to compute the RDP group size:
        rp_dim = floor(n_dims * rp_alpha / rp_group)
    dropout : float, default = 0.5
        dropout rate for the convolutional layers
    lstm_dropout : float, default = 0.8
        dropout rate for the LSTM layer
    dilation : int, default = 1
        dilation value
    padding : str, default = 'same'
        type of padding for convolution layers
    use_rp  : bool, default = True
        whether to use random projections
    use_att : bool, default = True
        whether to use self attention
    use_lstm : bool, default = True
        whether to use an LSTM layer
    use_cnn : bool, default = True
        whether to use a CNN layer
    fc_dropout : float, default = 0.0
        dropout rate before the output layer

    References
    ----------
    .. [1] Zhang et al. Tapnet: Multivariate time series classification with
    attentional prototypical network,
    Proceedings of the AAAI Conference on Artificial Intelligence
    34(4), 6845-6852, 2020
    """

    _tags = {
        "authors": ["jnrusson1", "noxthot"],
        "python_dependencies": ["tensorflow"],
    }

    def __init__(
        self,
        dropout=0.5,
        filter_sizes=(256, 256, 128),
        kernel_size=(8, 5, 3),
        dilation=1,
        layers=(500, 300),
        use_rp=True,
        rp_group=3,
        rp_alpha=2.0,
        use_att=True,
        use_lstm=True,
        use_cnn=True,
        random_state=None,
        padding="same",
        activation_hidden="leaky_relu",
        lstm_dropout=0.8,
        fc_dropout=0.0,
    ):
        _check_dl_dependencies(severity="error")

        super().__init__()

        self.activation_hidden = activation_hidden
        self.random_state = random_state
        self.kernel_size = kernel_size
        self.layers = layers
        self.filter_sizes = filter_sizes
        self.use_att = use_att
        self.dilation = dilation
        self.padding = padding

        self.dropout = dropout
        self.lstm_dropout = lstm_dropout
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.fc_dropout = fc_dropout

        self.use_rp = use_rp
        self.rp_group = rp_group
        self.rp_alpha = rp_alpha
        self._rng = np.random.default_rng(self.random_state)

        if not isinstance(self.kernel_size, tuple) or not all(
            isinstance(k, int) for k in self.kernel_size
        ):
            raise TypeError("`kernel_size` must be a tuple of ints.")

        if not isinstance(self.filter_sizes, tuple) or not all(
            isinstance(f, int) for f in self.filter_sizes
        ):
            raise TypeError("`filter_sizes` must be a tuple of ints.")

        if len(self.kernel_size) != len(self.filter_sizes):
            raise ValueError(
                "`kernel_size` and `filter_sizes` must be of the same length."
            )
        if len(self.kernel_size) < 1:
            raise ValueError("`kernel_size` and `filter_sizes` must have length >= 1.")

        if not isinstance(self.layers, tuple) or not all(
            isinstance(layer, int) for layer in self.layers
        ):
            raise TypeError("`layers` must be a tuple of ints.")
        if len(self.layers) < 1:
            raise ValueError("`layers` must have length >= 1.")

        if not isinstance(self.rp_group, int) or self.rp_group < 1:
            raise ValueError("`rp_group` must be a positive integer.")
        if not isinstance(self.rp_alpha, (int, float)) or self.rp_alpha <= 0:
            raise ValueError("`rp_alpha` must be a positive number.")

        if not self.use_lstm and not self.use_cnn:
            raise ValueError("At least one of `use_lstm` or `use_cnn` must be True.")

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
        from tensorflow import keras

        from sktime.libs._keras_self_attention import SeqSelfAttention

        input_layer = keras.layers.Input(input_shape)
        n_dims = input_shape[1]

        if self.use_rp:
            self.rp_dim = math.floor(n_dims * self.rp_alpha / self.rp_group)
            if self.rp_dim < 1:
                warnings.warn(
                    "Disabling random dimension permutation (RDP) because the "
                    "computed rp_dim is 0. This can happen for univariate data; "
                    "RDP requires multivariate inputs.",
                    UserWarning,
                )
                self.use_rp = False
                self.rp_dim = 0
            else:
                self.rp_dim = min(self.rp_dim, n_dims)
        else:
            self.rp_dim = 0

        def _apply_conv_block(x, conv_layer, bn_layer, activation_layer, dropout_layer):
            x = conv_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)
            if dropout_layer is not None:
                x = dropout_layer(x)
            return x

        if self.use_lstm:
            self.lstm_dim = 128

            x_lstm = keras.layers.LSTM(self.lstm_dim, return_sequences=True)(
                input_layer
            )
            x_lstm = keras.layers.Dropout(self.lstm_dropout)(x_lstm)

            if self.use_att:
                x_lstm = SeqSelfAttention(
                    self.lstm_dim, attention_type="multiplicative"
                )(x_lstm)
            x_lstm = keras.layers.GlobalAveragePooling1D()(x_lstm)

        if self.use_cnn:
            if self.use_rp:
                shared_blocks = []
                for i in range(1, len(self.filter_sizes)):
                    shared_blocks.append(
                        (
                            keras.layers.Conv1D(
                                self.filter_sizes[i],
                                kernel_size=self.kernel_size[i],
                                dilation_rate=self.dilation,
                                strides=1,
                                padding=self.padding,
                            ),
                            keras.layers.BatchNormalization(),
                            keras.layers.Activation(self.activation_hidden),
                            (
                                keras.layers.Dropout(self.dropout)
                                if self.dropout > 0.0
                                else None
                            ),
                        )
                    )

                rp_outputs = []
                for i in range(self.rp_group):
                    idx = self._rng.permutation(n_dims)[: self.rp_dim]
                    channel = keras.layers.Lambda(
                        lambda x, idx=idx: tf.gather(x, indices=idx, axis=2)
                    )(input_layer)
                    x_conv = _apply_conv_block(
                        channel,
                        keras.layers.Conv1D(
                            self.filter_sizes[0],
                            kernel_size=self.kernel_size[0],
                            dilation_rate=self.dilation,
                            strides=1,
                            padding=self.padding,
                        ),
                        keras.layers.BatchNormalization(),
                        keras.layers.Activation(self.activation_hidden),
                        (
                            keras.layers.Dropout(self.dropout)
                            if self.dropout > 0.0
                            else None
                        ),
                    )

                    for (
                        conv_layer,
                        bn_layer,
                        activation_layer,
                        dropout_layer,
                    ) in shared_blocks:
                        x_conv = _apply_conv_block(
                            x_conv,
                            conv_layer,
                            bn_layer,
                            activation_layer,
                            dropout_layer,
                        )

                    if self.use_att:
                        x_conv = SeqSelfAttention(
                            self.filter_sizes[-1], attention_type="multiplicative"
                        )(x_conv)

                    x_conv = keras.layers.GlobalAveragePooling1D()(x_conv)
                    rp_outputs.append(x_conv)

                x_conv = (
                    rp_outputs[0]
                    if len(rp_outputs) == 1
                    else keras.layers.Concatenate()(rp_outputs)
                )

            else:
                x_conv = _apply_conv_block(
                    input_layer,
                    keras.layers.Conv1D(
                        self.filter_sizes[0],
                        kernel_size=self.kernel_size[0],
                        dilation_rate=self.dilation,
                        strides=1,
                        padding=self.padding,
                    ),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation(self.activation_hidden),
                    keras.layers.Dropout(self.dropout) if self.dropout > 0.0 else None,
                )

                for i in range(1, len(self.filter_sizes)):
                    x_conv = _apply_conv_block(
                        x_conv,
                        keras.layers.Conv1D(
                            self.filter_sizes[i],
                            kernel_size=self.kernel_size[i],
                            dilation_rate=self.dilation,
                            strides=1,
                            padding=self.padding,
                        ),
                        keras.layers.BatchNormalization(),
                        keras.layers.Activation(self.activation_hidden),
                        (
                            keras.layers.Dropout(self.dropout)
                            if self.dropout > 0.0
                            else None
                        ),
                    )

                if self.use_att:
                    x_conv = SeqSelfAttention(self.filter_sizes[-1])(x_conv)

                x_conv = keras.layers.GlobalAveragePooling1D()(x_conv)

        if self.use_lstm and self.use_cnn:
            x = keras.layers.Concatenate()([x_conv, x_lstm])
        elif self.use_lstm:
            x = x_lstm
        else:
            x = x_conv

        for i, units in enumerate(self.layers):
            x = keras.layers.Dense(units)(x)
            if i < len(self.layers) - 1:
                x = keras.layers.Activation(self.activation_hidden)(x)
                x = keras.layers.BatchNormalization()(x)

        if self.fc_dropout > 0.0:
            x = keras.layers.Dropout(self.fc_dropout)(x)

        return input_layer, x
