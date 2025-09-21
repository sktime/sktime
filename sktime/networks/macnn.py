"""Multi-scale Attention Convolutional Neural Network (MACNN)."""

__author__ = ["jnrusson1"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class MACNNNetwork(BaseDeepNetwork):
    """Base MACNN Network for MACNNClassifier and MACNNRegressor.

    Parameters
    ----------
    padding : str, optional (default="same")
        The type of padding to be provided in MACNN Blocks. Accepts
        all the string values that keras.layers supports.
    pool_size : int, optional (default=3)
        A single value representing pooling windows which are applied
        between two MACNN Blocks.
    strides : int, optional (default=2)
        A single value representing strides to be taken during the
        pooling operation.
    repeats : int, optional (default=2)
        The number of MACNN Blocks to be stacked.
    filter_sizes : tuple, optional (default=(64, 128, 256))
        The input size of Conv1D layers within each MACNN Block.
    kernel_size : tuple, optional (default=(3, 6, 12))
        The output size of Conv1D layers within each MACNN Block.
    reduction : int, optional (default=16)
        The factor by which the first dense layer of a MACNN Block will be divided by.
    random_state: int, optional (default=0)
        The seed to any random action.
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        padding="same",
        pool_size=3,
        strides=2,
        repeats=2,
        filter_sizes=(64, 128, 256),
        kernel_size=(3, 6, 12),
        reduction=16,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")
        super().__init__()

        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides
        self.repeats = repeats
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.random_state = random_state

    def _macnn_block(self, x, kernels, reduce):
        """Implement a single MACNN Block.

        Parameters
        ----------
        x : An instance of keras.layers.Layer
            The previous layer, in case of the first
            block it represents the input layer.
        kernels: int
            The base output dimension for dense layers, it corresponds
            to elements `filter_sizes` attributes.
        reduce: int
            The factor by which, the first dense layer's output dimension
            should be divided by, it corresponds to the `reduction` attribute.

        Returns
        -------
        block_output: An instance of keras.layers.Layer
            Represents the last layer of a MACNN Block, to be used by the next block.
        """
        from tensorflow import keras

        conv_layers = []
        for kernel_size in self.kernel_size:
            conv_layer = keras.layers.Conv1D(
                filters=kernels, kernel_size=kernel_size, padding=self.padding
            )(x)

            conv_layers.append(conv_layer)

        x1 = keras.layers.Concatenate(axis=2)(conv_layers)
        x1 = keras.layers.BatchNormalization()(x1)
        x1 = keras.layers.Activation("relu")(x1)

        x2 = keras.layers.GlobalAveragePooling1D()(x1)
        x2 = keras.layers.Dense(
            units=int(kernels * 3 / reduce), use_bias=False, activation="relu"
        )(x2)
        x2 = keras.layers.Dense(
            units=int(kernels * 3), use_bias=False, activation="relu"
        )(x2)
        x2 = keras.layers.RepeatVector(x1.shape[1])(x2)

        return keras.layers.Multiply()([x1, x2])

    def _stack(self, x, repeats, kernels, reduce):
        """Build MACNN Blocks and stack them.

        Parameters
        ----------
        x : Instance of keras.layers.Layer
            The previous layer, in case of the first
            block it represents the input layer.
        repeats : int
            The number of MACNN Blocks to be used.
            Corresponds to `repeats` attribute.
        kernels : int
            The base output dimension for dense layers, it corresponds
            to elements `filter_sizes` attributes.

        Returns
        -------
        x : Instance of keras.layers.Layer
            The final layer after repeatedly applying MACNN Blocks.
        """
        for _ in range(repeats):
            x = self._macnn_block(x, kernels, reduce)
        return x

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data (without batch) fed into the input layer

        Returns
        -------
        input_layer : An instance of keras.layers.Input
            The input layer of this Network.
        output_layer: An instance of keras.layers.Layer
            The output layer of this Network.
        """
        from tensorflow import keras

        input_layer = keras.layers.Input(shape=input_shape)

        x = self._stack(
            x=input_layer,
            repeats=self.repeats,
            kernels=self.filter_sizes[0],
            reduce=self.reduction,
        )
        x = keras.layers.MaxPooling1D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
        )(x)

        x = self._stack(
            x=x,
            repeats=self.repeats,
            kernels=self.filter_sizes[1],
            reduce=self.reduction,
        )
        x = keras.layers.MaxPooling1D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
        )(x)

        x = self._stack(
            x=x,
            repeats=self.repeats,
            kernels=self.filter_sizes[2],
            reduce=self.reduction,
        )

        output_layer = keras.layers.GlobalAveragePooling1D()(x)

        return input_layer, output_layer

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                    should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {
            "padding": "valid",
            "pool_size": 2,
            "strides": 1,
            "repeats": 2,
            "filter_sizes": (64, 128, 256),
            "kernel_size": (3, 6, 12),
            "reduction": 16,
            "random_state": 0,
        }
        return [params1, params2]
