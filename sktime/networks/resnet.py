"""Residual Network (ResNet) (minus the final output layer)."""

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class ResNetNetwork(BaseDeepNetwork):
    """Establish the network structure for a ResNet.

    Adapted from the implementation in
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    Parameters
    ----------
    random_state : int, optional (default = 0)
        The random seed to use random activities.
    activation : string, optional (default = "relu")
        Activation function used for hidden layers;
        List of available keras activation functions:
        https://keras.io/api/layers/activations/
    n_filters : tuple of int, optional (default = (64, 128, 128))
        Number of filters for each residual block. The length of the tuple
        determines the number of residual blocks.
    kernel_sizes : tuple of int, default = (8, 5, 3)
        Kernel sizes for the three conv layers in each residual block.
    padding : str, default = "same"
        Padding type for all conv layers.

    References
    ----------
    .. [1] H. Fawaz, G. B. Lanckriet, F. Petitjean, and L. Idoumghar,

    Network originally defined in:

    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }
    """

    _tags = {
        "authors": ["hfawaz", "James-Large", "Withington", "nilesh05apr", "noxthot"],
        "python_dependencies": ["tensorflow"],
    }

    def __init__(
        self,
        random_state=0,
        activation="relu",
        n_filters=(64, 128, 128),
        kernel_sizes=(8, 5, 3),
        padding="same",
    ):
        _check_dl_dependencies(severity="error")
        super().__init__()
        self.random_state = random_state
        self.activation = activation
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.padding = padding

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple of shape = (series_length (m), n_dimensions (d))
            The shape of the data fed into the input layer.

        Returns
        -------
        input_layer : keras.layers.Input
            The input layer of the network.
        output_layer : keras.layers.Layer
            The output layer of the network.
        """
        from tensorflow import keras

        input_layer = keras.layers.Input(input_shape)
        x = input_layer

        # stack residual blocks; number of blocks = len(self.n_filters)
        for filters in self.n_filters:

            # conv block: 3 conv layers with kernel sizes from self.kernel_sizes
            conv_x = keras.layers.Conv1D(
                filters=filters, kernel_size=self.kernel_sizes[0], padding=self.padding
            )(x)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation(self.activation)(conv_x)

            conv_y = keras.layers.Conv1D(
                filters=filters, kernel_size=self.kernel_sizes[1], padding=self.padding
            )(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation(self.activation)(conv_y)

            conv_z = keras.layers.Conv1D(
                filters=filters, kernel_size=self.kernel_sizes[2], padding=self.padding
            )(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)

            # shortcut: expand channels if filter count changed, else batch norm only
            if int(x.shape[-1]) == filters:
                shortcut = keras.layers.BatchNormalization()(x)
            else:
                shortcut = keras.layers.Conv1D(
                    filters=filters, kernel_size=1, padding=self.padding
                )(x)
                shortcut = keras.layers.BatchNormalization()(shortcut)

            x = keras.layers.add([shortcut, conv_z])
            x = keras.layers.Activation(self.activation)(x)

        # global average pooling
        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        return input_layer, gap_layer

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
        params2 = {"random_state": 42, "n_filters": (32, 64)}
        return [params1, params2]
