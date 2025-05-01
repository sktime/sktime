"""a combination of CCNN and CLSTM as feature extractors."""

__author__ = ["James-Large", "Withington", "TonyBagnall", "AurumnPegasus"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class CNTCNetwork(BaseDeepNetwork):
    """Combining contextual neural networks for time series classification.

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
    Adapted from the implementation from Fullah et. al
    https://github.com/AmaduFullah/CNTC_MODEL/blob/master/cntc.ipynb

    References
    ----------
    .. [1] Network originally defined in:
        @article{FULLAHKAMARA202057,
        title = {Combining contextual neural networks for time series classification},
        journal = {Neurocomputing},
        volume = {384},
        pages = {57-66},
        year = {2020},
        issn = {0925-2312},
        doi = {https://doi.org/10.1016/j.neucom.2019.10.113},
        url = {https://www.sciencedirect.com/science/article/pii/S0925231219316364},
        author = {Amadu {Fullah Kamara} and Enhong Chen and Qi Liu and Zhen Pan},
        keywords = {Time series classification, Contextual convolutional neural
            networks, Contextual long short-term memory, Attention, Multilayer
            perceptron},
       }
    """

    _tags = {
        "authors": ["James-Large", "Withington", "TonyBagnall", "AurumnPegasus"],
        "maintainers": ["James-Large", "Withington", "AurumnPegasus"],
        "python_dependencies": ["tensorflow"],
    }

    def __init__(
        self,
        random_state=0,
        rnn_layer=64,
        filter_sizes=(16, 8),
        kernel_sizes=(1, 1),
        lstm_size=8,
        dense_size=64,
    ):
        _check_dl_dependencies(severity="error")

        self.random_state = random_state
        self.rnn_layer = rnn_layer
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.lstm_size = lstm_size
        self.dense_size = dense_size

        super().__init__()

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape: tuple
            The shape of the data fed into the input layer, should be (m,d)

        Returns
        -------
        input_layer: a keras layer
        output_layer: a keras layer
        """
        from tensorflow import keras

        from sktime.libs._keras_self_attention import SeqSelfAttention

        input_layers = []

        # CNN Arm
        input_layers.append(keras.layers.Input(input_shape))
        input_layers.append(keras.layers.Input(input_shape))
        self.dropout = 0.2

        conv1 = keras.layers.Conv1D(
            self.filter_sizes[0],
            self.kernel_sizes[0],
            activation="relu",
            use_bias=True,
            kernel_initializer="glorot_uniform",
        )(input_layers[0])
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Dropout(self.dropout)(conv1)
        conv1 = keras.layers.Dense(
            input_shape[1],
            input_shape=(input_shape[0], keras.backend.int_shape(conv1)[2]),
        )(conv1)

        # RNN for CNN Arm (CCNN)
        rnn1 = keras.layers.SimpleRNN(
            self.rnn_layer * input_shape[1],
            activation="relu",
            use_bias=True,
            kernel_initializer="glorot_uniform",
        )(input_layers[1])
        rnn1 = keras.layers.BatchNormalization()(rnn1)
        rnn1 = keras.layers.Dropout(self.dropout)(rnn1)
        rnn1 = keras.layers.Reshape((64, input_shape[1]))(rnn1)

        # Combining CNN and RNN
        conc1 = keras.layers.Concatenate(
            axis=-2, name="contextual_convolutional_layer1"
        )([conv1, rnn1])

        # Final CNN for C-CNN (WHY)
        conv2 = keras.layers.Conv1D(
            self.filter_sizes[1],
            self.kernel_sizes[1],
            activation="relu",
            kernel_initializer="glorot_uniform",
            name="standard_cnn_layer",
        )(conc1)
        conv2 = keras.layers.Dense(
            input_shape[1],
            input_shape=(input_shape[0], keras.backend.int_shape(conv2)[2]),
        )(conv2)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Dropout(0.1)(conv2)

        # CLSTM Arm
        input_layers.append(keras.layers.Input(input_shape))
        lstm1 = keras.layers.LSTM(
            self.lstm_size * input_shape[1],
            return_sequences=False,
            kernel_initializer="glorot_uniform",
            activation="relu",
        )(input_layers[2])
        lstm1 = keras.layers.Reshape((self.lstm_size, input_shape[1]))(lstm1)
        lstm1 = keras.layers.Dropout(self.dropout)(lstm1)
        merge = keras.layers.Concatenate(
            axis=-2, name="contextual_convolutional_layer2"
        )([conv2, lstm1])

        # Output calculation based on combination
        avg = keras.layers.MaxPooling1D(pool_size=1, strides=None, padding="valid")(
            merge
        )
        avg = keras.layers.Dropout(0.1)(avg)

        # Adding self attention
        att = SeqSelfAttention(
            attention_width=10,
            attention_activation="sigmoid",
            name="Attention",
            attention_type="multiplicative",
        )(avg)
        att = keras.layers.Dropout(0.1)(att)

        # Adding ouutput MLP Layer
        mlp1 = keras.layers.Dense(
            self.dense_size, kernel_initializer="glorot_uniform", activation="relu"
        )(att)
        mlp1 = keras.layers.Dropout(0.1)(mlp1)
        mlp2 = keras.layers.Dense(
            self.dense_size, kernel_initializer="glorot_uniform", activation="relu"
        )(mlp1)
        mlp2 = keras.layers.Dropout(0.1)(mlp2)
        flat = keras.layers.Flatten()(mlp2)
        return input_layers, flat
