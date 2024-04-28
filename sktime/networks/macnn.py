"""Multi-scale Attention Convolutional Neural Network (MACNN)."""

__author__ = ["jnrusson1"]

from tensorflow import keras
from tensorflow.keras import layers

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies


# Wrapper Layers for using KerasTensor in tf func.
class ReduceMean(layers.Layer):
    """Wrapper layer to apply mean reduction along a specified axis.

    This layer computes the mean of the input tensor along the specified axis.

    Parameters
    ----------
    axis : int, optional (default=1)
        The axis along which to compute the mean.
    **kwargs
        Additional keyword arguments passed to the base `layers.Layer` class.
    """

    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        """Apply mean reduction along the specified axis.

        Parameters
        ----------
        inputs : tensor (input tensor)
            The input tensor to apply mean reduction.
        Returns: tensor
            The tensor after applying mean reduction along the specified axis.
        """
        return keras.backend.mean(inputs, axis=self.axis)


class Reshape(layers.Layer):
    """Wrapper layer to reshape the input tensor.

    This layer reshapes the input tensor to the specified shape.

    Parameters
    ----------
    shape : tuple or list
        The target shape for the input tensor.
    **kwargs
        Additional keyword arguments passed to the base `layers.Layer` class.
    """

    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def call(self, inputs):
        """Reshape the input tensor.

        Parameters
        ----------
        inputs : tensor (input tensor)
            The input tensor to apply reshape.
        Returns: tensor
        """
        return keras.backend.reshape(inputs, self.shape)


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
        for i in range(len(self.kernel_size)):
            conv_layers.append(
                keras.layers.Conv1D(kernels, self.kernel_size[i], padding=self.padding)(
                    x
                )
            )

        x1 = keras.layers.Concatenate(axis=2)(conv_layers)
        x1 = keras.layers.BatchNormalization()(x1)
        x1 = keras.layers.Activation("relu")(x1)

        x2 = ReduceMean(axis=1)(x1)
        x2 = keras.layers.Dense(
            int(kernels * 3 / reduce), use_bias=False, activation="relu"
        )(x2)
        x2 = keras.layers.Dense(int(kernels * 3), use_bias=False, activation="relu")(x2)
        x2 = Reshape([-1, 1, kernels * 3])(x2)

        return x1 * x2

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
            The shape of the data(without batch) fed into the input layer

        Returns
        -------
        input_layer : An instance of keras.layers.Input
            The input layer of this Network.
        output_layer: An instance of keras.layers.Layer
            The output layer of this Network.
        """
        from tensorflow import keras

        input_layer = keras.layers.Input(shape=input_shape)

        x = self._stack(input_layer, self.repeats, self.filter_sizes[0], self.reduction)
        x = keras.layers.MaxPooling1D(
            self.pool_size, self.strides, padding=self.padding
        )(x)

        x = self._stack(x, self.repeats, self.filter_sizes[1], self.reduction)
        x = keras.layers.MaxPooling1D(
            self.pool_size, self.strides, padding=self.padding
        )(x)

        x = self._stack(x, self.repeats, self.filter_sizes[2], self.reduction)

        output_layer = ReduceMean(axis=1)(x)

        return input_layer, output_layer
