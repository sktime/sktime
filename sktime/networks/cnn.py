"""Time Convolutional Neural Network (CNN) (minus the final output layer)."""

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_dl_dependencies
from sktime.utils.warnings import warn


# TODO (release 0.42.0)
# change the default value of 'activation' to "relu"
# update the docstring for activation from "sigmoid" to "relu"
# and remove the note about the change from the docstring.
# Remove the usage of self._activation throughout the class
# and replace it with self.activation
class CNNNetwork(BaseDeepNetwork):
    """Establish the network structure for a CNN.

    Zhao et al. 2017 uses sigmoid activation in the hidden layers.
    To obtain same behaviour as Zhao et al. 2017, set activation to "sigmoid".

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    kernel_size : int, default = 7
        specifying the length of the 1D convolution window
    avg_pool_size : int, default = 3
        size of the average pooling windows
    n_conv_layers : int, default = 2
        the number of convolutional plus average pooling layers
    filter_sizes : array of int, shape = (n_conv_layers)
    activation : string, default = "sigmoid"
        activation function used for hidden layers;
        List of available keras activation functions:
        https://keras.io/api/layers/activations/
        Default value of activation will change to "relu"
        in version '0.41.0'.
    padding : string, default = "auto"
        Controls padding logic for the convolutional layers,
        i.e. whether ``'valid'`` and ``'same'`` are passed to the ``Conv1D`` layer.
        - "auto": as per original implementation, ``"same"`` is passed if
          ``input_shape[0] < 60`` in the input layer, and ``"valid"`` otherwise.
        - "valid", "same", and other values are passed directly to ``Conv1D``
    random_state    : int, default = 0
        seed to any needed random actions

    References
    ----------
    .. [1] Zhao et al. Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics 28(1), 162--169, 2017
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["hfawaz", "James-Large", "Withington", "TonyBagnall", "noxthot"],
        "python_dependencies": "tensorflow",
    }

    # TODO (release 0.42.0)
    # Change the default value of 'activation' to "relu"
    def __init__(
        self,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        filter_sizes=None,
        activation="changing_from_sigmoid_to_relu_in_0.41.0",
        padding="auto",
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")
        self.random_state = random_state
        self.padding = padding
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = filter_sizes
        if filter_sizes is None:
            self._filter_sizes = [6, 12]
        else:
            self._filter_sizes = filter_sizes
        self.activation = activation
        # TODO (release 0.42.0)
        # After changing the default value of 'activation' to "relu"
        # in the __init__ method signature,
        # Remove the following 'if-else' check
        # Remove the usage of self._activation throughout the class
        # and replace it with self.activation
        if activation == "changing_from_sigmoid_to_relu_in_0.41.0":
            warn(
                "in `CNNNetwork`, the default value of parameter 'activation'"
                " will change to 'relu' in version '0.41.0'. "
                "To keep current behaviour and to silence this warning, "
                "set 'activation' to 'sigmoid' explicitly.",
                category=DeprecationWarning,
                obj=self,
            )
            self._activation = "sigmoid"
        else:
            self._activation = activation

        super().__init__()

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

        fs = self._filter_sizes
        nconv = self.n_conv_layers
        padding = self.padding

        # this is the condition from the original implementation
        if padding == "auto":
            if input_shape[0] < 60:
                padding = "same"
            else:
                padding = "valid"

        # Extends filter_sizes to match n_conv_layers length
        filter_sizes = fs[:nconv] + [fs[-1]] * max(0, nconv - len(fs))

        input_layer = keras.layers.Input(input_shape)

        # TODO (release 0.42.0)
        # After changing the default value of 'activation' to "relu"
        # in the __init__ method signature,
        # remove the usage of self._activation in the following lines
        # and replace it with self.activation
        conv = keras.layers.Conv1D(
            filters=filter_sizes[0],
            kernel_size=self.kernel_size,
            padding=padding,
            activation=self._activation,
        )(input_layer)
        conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(conv)

        for i in range(1, self.n_conv_layers):
            # TODO (release 0.42.0)
            # After changing the default value of 'activation' to "relu"
            # in the __init__ method signature,
            # remove the usage of self._activation in the following lines
            # and replace it with self.activation
            conv = keras.layers.Conv1D(
                filters=filter_sizes[i],
                kernel_size=self.kernel_size,
                padding=padding,
                activation=self._activation,
            )(conv)
            conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(conv)

        flatten_layer = keras.layers.Flatten()(conv)

        return input_layer, flatten_layer

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
            "filter_sizes": [10],
            "kernel_size": 3,
            "avg_pool_size": 1,
            "padding": "valid",
            "n_conv_layers": 1,
        }
        params3 = {"kernel_size": 5, "padding": "same"}
        return [params1, params2, params3]
