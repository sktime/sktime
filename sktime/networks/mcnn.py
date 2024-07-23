"""Multi-scale Convolutional Neural Network."""

__author__ = ["fnhirwa"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class MCNNNetwork(BaseDeepNetwork):
    """Base MCNN Network for MCNNClassifier and MCNNRegressor.

    Network is originally defined in [1]_.

    Parameters
    ----------
    input_shapes: tuple
        The shapes of three input branches.
    pool_factor: int, optional (default=2)
        The factor by which the pooling size is divided.
    kernel_size: int, optional (default=7)
        The size of the 1D convolutional window.
    padding: str, optional (default="same")
        Controls padding logic for the convolutional layers,
        i.e. whether ``'valid'`` and ``'same'`` are passed to the ``Conv1D`` layer.
        - "auto": as per original implementation, ``"same"`` is passed if
          ``input_shape[0] < 60`` in the input layer, and ``"valid"`` otherwise.
        - "valid", "same", and other values are passed directly to ``Conv1D``
    random_state: int, optional (default=0)
        The seed to any random action.

    References
    ----------
    .. [1] Cui, Z., Chen, W., & Chen, Y. (2016). Multi-scale convolutional neural networks for time series classification. arXiv preprint arXiv:1603.06995.
    """  # noqa: E501

    _tags = {"python_dependencies": "tensorflow==2.15.0"}

    def __init__(
        self,
        pool_factor=2,
        kernel_size=7,
        padding="same",
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")

        self.pool_factor = pool_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.random_state = random_state

        super().__init__()

    def build_network(self, input_shape, **kwargs):
        """Build the MCNN network and return its input and output layers.

        Parameters
        ----------
        input_shape: tuple
            The shape of the input data.

        Returns
        -------
        tuple
            A tuple containing the input and fully connected layers of the network.
        """
        import tensorflow as tf

        input_layers = []
        first_stage_layers = []

        for i in range(len(input_shape)):
            input_layer = tf.keras.layers.Input(input_shape[i])
            input_layers.append(input_layer)

            conv_layer = tf.keras.layers.Conv1D(
                filters=256,
                kernel_size=self.kernel_size,
                padding=self.padding,
                activation="sigmoid",
                kernel_initializer="glorot_uniform",
            )(input_layer)

            # selecting the pooling size based on the convolutional layer output size
            pool_size = int(int(conv_layer.shape[1]) / self.pool_factor)

            pool_layer = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(conv_layer)
            first_stage_layers.append(pool_layer)

        # concatenating the three branches
        merged_layer = tf.keras.layers.Concatenate(axis=-1)(first_stage_layers)

        # kernel size as it shouldn't exceed the input size

        kernel_size = int(min(self.kernel_size, int(merged_layer.shape[1])))

        # full conv layer
        full_conv_layer = tf.keras.layers.Conv1D(
            filters=256,
            kernel_size=kernel_size,
            padding=self.padding,
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
        )(merged_layer)

        # pooling
        pool_size = int(int(full_conv_layer.shape[1]) / self.pool_factor)

        pool_layer = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(full_conv_layer)

        # flatten
        flatten_layer = tf.keras.layers.Flatten()(pool_layer)

        fully_connected_layer = tf.keras.layers.Dense(
            units=256,
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
        )(flatten_layer)
        return input_layers, fully_connected_layer

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
            "pool_factor": 3,
            "kernel_size": 5,
            "padding": "same",
            "random_state": 0,
        }
        parms3 = {
            "pool_factor": 2,
            "kernel_size": 7,
            "padding": "auto",
        }
        return [params1, params2, parms3]
