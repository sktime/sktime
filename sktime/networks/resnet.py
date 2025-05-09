"""Residual Network (ResNet) (minus the final output layer)."""

__author__ = ["James-Large", "Withington", "nilesh05apr"]

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
        "authors": ["hfawaz", "James-Large", "Withington", "nilesh05apr"],
        "python_dependencies": ["tensorflow"],
    }

    def __init__(self, random_state=0):
        _check_dl_dependencies(severity="error")
        super().__init__()
        self.random_state = random_state

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

        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # 1st residual block

        conv_x = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=8, padding="same"
        )(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=5, padding="same"
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=3, padding="same"
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=1, padding="same"
        )(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation("relu")(output_block_1)

        # 2nd residual block

        conv_x = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=8, padding="same"
        )(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=5, padding="same"
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=3, padding="same"
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=1, padding="same"
        )(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation("relu")(output_block_2)

        # 3rd residual block

        conv_x = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=8, padding="same"
        )(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=5, padding="same"
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=3, padding="same"
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation("relu")(output_block_3)

        # global average pooling

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

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
        params2 = {"random_state": 42}
        return [params1, params2]
