"""Fully Connected Neural Network (FCN) (minus the final output layer)."""

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class FCNNetwork(BaseDeepNetwork):
    """Establish the network structure for a FCN.

    Adapted from the implementation of Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py

    Implements network in [1]_.

    Parameters
    ----------
    random_state : int, default = 0
        seed to any needed random actions
    activation : string, default = "relu"
        activation function used for hidden layers;
        List of available keras activation functions:
        https://keras.io/api/layers/activations/
    filter_sizes : list or tuple of int, default = (128, 256, 128)
        number of filters for each convolutional layer.
        must have length equal to kernel_sizes.
    kernel_sizes : list or tuple of int, default = (8, 5, 3)
        kernel size for each convolutional layer.
        must have length equal to filter_sizes.

    References
    ----------
    .. [1] Network originally defined in:
    @inproceedings{wang2017time,
      title={Time series classification from scratch with deep neural networks:
       A strong baseline},
      author={Wang, Zhiguang and Yan, Weizhong and Oates, Tim},
      booktitle={2017 International joint conference on neural networks
      (IJCNN)},
      pages={1578--1585},
      year={2017},
      organization={IEEE}
    }
    """

    _tags = {
        "authors": ["James-Large", "AurumnPegasus", "noxthot"],
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        random_state=0,
        activation="relu",
        filter_sizes=(128, 256, 128),
        kernel_sizes=(8, 5, 3),
    ):
        super().__init__()
        _check_dl_dependencies(severity="error")
        self.random_state = random_state
        self.activation = activation
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes

        # type check for filter_sizes
        if not isinstance(self.filter_sizes, (list, tuple)):
            raise ValueError(
                f"filter_sizes must be a list or tuple ,"
                f"but got type {type(self.filter_sizes)}."
            )
        # type check for kernel_sizes
        if not isinstance(self.kernel_sizes, (list, tuple)):
            raise ValueError(
                f"kernel_sizes must be a list or tuple ,"
                f"but got type {type(self.kernel_sizes)}."
            )
        if len(self.filter_sizes) != len(self.kernel_sizes):
            raise ValueError(
                f"filter_sizes and kernel_sizes must have the same length ,"
                f"but got {len(self.filter_sizes)} and {len(self.kernel_sizes)}."
            )

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple of shape = (series_length (m), n_dimensions (d))
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        from tensorflow import keras

        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        for i in range(len(self.filter_sizes)):
            x = keras.layers.Conv1D(
                filters=self.filter_sizes[i],
                kernel_size=self.kernel_sizes[i],
                padding="same",
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation=self.activation)(x)

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
        params2 = {
            "random_state": 42,
            "filter_sizes": [64, 128],
            "kernel_sizes": [5, 3],
        }

        # for tuple checking
        params3 = {
            "random_state": 1,
            "filter_sizes": (64, 128),
            "kernel_sizes": (5, 3),
        }

        return [params1, params2, params3]
