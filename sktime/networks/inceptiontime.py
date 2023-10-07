"""Inception Time."""
__author__ = ["JamesLarge", "Withington"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies


class InceptionTimeNetwork(BaseDeepNetwork):
    """InceptionTime adapted from the implementation from Fawaz et al.

    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/
    inception.py

    Network originally defined in:

    @article{IsmailFawaz2019inceptionTime, Title                    = {
    InceptionTime: Finding AlexNet for Time Series Classification}, Author
                    = {Ismail Fawaz, Hassan and Lucas, Benjamin and
                    Forestier, Germain and Pelletier, Charlotte and Schmidt,
                    Daniel F. and Weber, Jonathan and Webb, Geoffrey I. and
                    Idoumghar, Lhassane and Muller, Pierre-Alain and
                    Petitjean, François}, journal                  = {
                    ArXiv}, Year                     = {2019} }
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        n_filters=32,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        kernel_size=40,
        random_state=0,
    ):
        """Initialize Inception Time.

        ----------
        :param n_filters: int,
        :param use_residual: boolean,
        :param use_bottleneck: boolean,
        :param depth: int
        :param kernel_size: int, specifying the length of the 1D convolution
         window
        :param bottleneck_size: int,
        :param random_state: int, seed to any needed random actions
        """
        _check_dl_dependencies(severity="error")
        super().__init__()

        self.n_filters = n_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size
        self.bottleneck_size = bottleneck_size
        self.random_state = random_state

    def _inception_module(self, input_tensor, stride=1, activation="linear"):
        from tensorflow import keras

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                padding="same",
                activation=activation,
                use_bias=False,
            )(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2**i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                keras.layers.Conv1D(
                    filters=self.n_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )

        max_pool_1 = keras.layers.MaxPool1D(
            pool_size=3, strides=stride, padding="same"
        )(input_tensor)

        conv_6 = keras.layers.Conv1D(
            filters=self.n_filters,
            kernel_size=1,
            padding="same",
            activation=activation,
            use_bias=False,
        )(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation="relu")(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        from tensorflow import keras

        shortcut_y = keras.layers.Conv1D(
            filters=int(out_tensor.shape[-1]),
            kernel_size=1,
            padding="same",
            use_bias=False,
        )(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation("relu")(x)
        return x

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        from tensorflow import keras

        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        return input_layer, gap_layer
