"""LongShort Term Memory Fully Convolutional Network (LSTM-FCN)."""

from sktime.networks.base import BaseDeepNetwork


class LSTMFCNNetwork(BaseDeepNetwork):
    """Implementation of LSTMFCNClassifier from Karim et al (2019) [1]_.

    Overview
    --------
    Combines an LSTM arm with a CNN arm. Optionally uses an attention mechanism in the
    LSTM which the author indicates provides improved performance.

    Parameters
    ----------
    kernel_sizes : list or tuple of int, default=(8, 5, 3)
        Length of the 1D convolution windows for each convolutional layer.
        The number of convolutional layers is ``len(kernel_sizes)``.
        Must have the same length as ``filter_sizes``.
        Defaults match Karim et al. (2019): three layers with kernels 8, 5, 3.
    filter_sizes : list or tuple of int, default=(128, 256, 128)
        Number of filters for each convolutional layer.
        The number of convolutional layers is ``len(filter_sizes)``.
        Must have the same length as ``kernel_sizes``.
        Defaults match Karim et al. (2019): three layers with 128, 256, 128 filters.
    random_state : int, default=0
        Seed for any needed random actions.
    lstm_size : int, default=8
        Output dimension for LSTM layer.
    dropout : float, default=0.8
        Dropout rate of LSTM layer.
    attention : bool, default=False
        If True, uses custom attention LSTM layer.
    activation : str, default="relu"
        Activation function used for hidden layers.
        List of available keras activation functions:
        https://keras.io/api/layers/activations/

    Notes
    -----
    Ported from sktime-dl source code
    https://github.com/sktime/sktime-dl/blob/master/sktime_dl/networks/_lstmfcn.py

    References
    ----------
    .. [1] Karim et al. Multivariate LSTM-FCNs for Time Series Classification, 2019
       https://arxiv.org/pdf/1801.04503.pdf

    Examples
    --------
    >>> from sktime.networks.lstmfcn import LSTMFCNNetwork
    >>> network = LSTMFCNNetwork(
    ...     kernel_sizes=(5, 3), filter_sizes=(64, 128), random_state=42
    ... )
    """

    _tags = {
        "authors": ["jnrusson1", "solen0id", "noxthot"],
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        kernel_sizes=(8, 5, 3),
        filter_sizes=(128, 256, 128),
        random_state=0,
        lstm_size=8,
        dropout=0.8,
        attention=False,
        activation="relu",
    ):
        """Initialize a new LSTMFCNNetwork object."""
        self.activation = activation
        self.random_state = random_state
        self.kernel_sizes = kernel_sizes
        self.filter_sizes = filter_sizes
        self.lstm_size = lstm_size
        self.dropout = dropout
        self.attention = attention

        super().__init__()

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        if not isinstance(self.filter_sizes, (list, tuple)):
            raise ValueError(
                f"filter_sizes must be a list or tuple, "
                f"but got type {type(self.filter_sizes)}."
            )
        if not isinstance(self.kernel_sizes, (list, tuple)):
            raise ValueError(
                f"kernel_sizes must be a list or tuple, "
                f"but got type {type(self.kernel_sizes)}."
            )
        if len(self.filter_sizes) != len(self.kernel_sizes):
            raise ValueError(
                f"filter_sizes and kernel_sizes must have the same length, "
                f"but got {len(self.filter_sizes)} and {len(self.kernel_sizes)}."
            )

        super().__post_init__()

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

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

        input_layer = keras.layers.Input(shape=input_shape)

        if self.attention:
            x = keras.layers.Attention()([input_layer, input_layer])
        else:
            x = input_layer

        x = keras.layers.LSTM(self.lstm_size)(x)
        x = keras.layers.Dropout(self.dropout)(x)

        y = input_layer
        for i in range(len(self.filter_sizes)):
            y = keras.layers.Conv1D(
                self.filter_sizes[i],
                self.kernel_sizes[i],
                padding="same",
                kernel_initializer="he_uniform",
            )(y)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Activation(self.activation)(y)

        y = keras.layers.GlobalAveragePooling1D()(y)

        output_layer = keras.layers.concatenate([x, y])

        return input_layer, output_layer

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            # Advanced model version
            {
                "kernel_sizes": (8, 5, 3),  # Keep standard kernel sizes
                "filter_sizes": (128, 256, 128),  # Keep standard kernel counts
                "lstm_size": 8,
                "dropout": 0.25,  # Maintain lower dropout rate for attention model
                "attention": True,
            },
            # Simpler model version
            {
                "kernel_sizes": (4, 2, 1),  # Reduce kernel sizes
                "filter_sizes": (32, 64, 32),  # Reduc filter sizes for cheaper model
                "lstm_size": 8,  # Keeping LSTM output size fixed
                "dropout": 0.75,  # Maintain higher dropout rate for non attention model
                "attention": False,
            },
            {},
            # Dynamic number of conv layers via list inputs
            {
                "kernel_sizes": [5, 3],
                "filter_sizes": [64, 128],
            },
            # Dynamic number of conv layers via tuple inputs
            {
                "kernel_sizes": (5, 3),
                "filter_sizes": (64, 128),
            },
        ]

        return params
