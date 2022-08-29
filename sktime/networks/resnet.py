"""ResNet (minus the final output layer)."""
__author__ = ["James Large", "Withington", "Nilesh"]

from tensorflow import keras
from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")

class ResNetNetwork(BaseDeepNetwork):
    """Residual Network (ResNet).

    Adapted from the implementation used in [1]

    Parameters
    ----------
    random_state    : int, default = 0
        seed to any needed random actions

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py
    
    Refrences
    ---------
    .. [1] Network originally defined in:
    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self, 
        random_state=0
    ):
        _check_dl_dependencies(severity="error")
        self.random_state = random_state
        super(ResNetNetwork, self).__init__()

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

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

        # BLOCK 2

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

        # BLOCK 3

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

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        return input_layer, gap_layer

