# -*- coding: utf-8 -*-
import inspect
from typing import Tuple

import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, layers
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GlobalMaxPool1D,
    Input,
    Layer,
    LeakyReLU,
    Reshape,
)

from sktime.clustering.dnn.encoders._base import BaseEncoder


class _CausalResidualBlock(Layer):
    def __init__(
        self,
        nb_filters,
        kernel_size,
        dilation_rate,
        final=False,
        kernel_initializer=None,
        **kwargs
    ):
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        if self.kernel_initializer is None:
            self.kernel_initializer = initializers.VarianceScaling(
                (1.0 / 3), distribution="uniform"
            )
        self.final = final
        self.layers_outputs = []
        self.layers = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(_CausalResidualBlock, self).__init__(**kwargs)

    def build(self, input_shape):

        self.final = input_shape[-1] != self.nb_filters
        with K.name_scope(
            self.name
        ):  # name scope used to make sure weights get unique names
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = "conv1D_{}".format(k)
                with K.name_scope(name):
                    self._add_and_activate_layer(
                        tfa.layers.WeightNormalization(
                            Conv1D(
                                filters=self.nb_filters,
                                kernel_size=self.kernel_size,
                                dilation_rate=self.dilation_rate,
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.kernel_initializer,
                                padding="causal",
                                name=name,
                            ),
                            data_init=False,
                        )
                    )
                self._add_and_activate_layer(LeakyReLU(alpha=0.01))

            if self.final:
                # 1x1 conv to match the shapes (channel dimension).
                name = "conv1D_3"
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = Conv1D(
                        filters=self.nb_filters,
                        kernel_size=1,
                        kernel_initializer=self.kernel_initializer,
                        padding="same",
                        name=name,
                    )
                # else:
                #     self.shape_match_conv = Lambda(lambda x: x, name='identity')
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(
                    input_shape
                )

            # if self.final:
            #     name = 'activ'
            #     with K.name_scope(name):
            #         self.final_activation = Activation('relu')
            #         # self.final_activation = Activation(LeakyReLU(alpha=0.01))
            #         self.final_activation.build(self.res_output_shape)  # probably isn't necessary
            # else:
            #     self.final_activation = None

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)

            super(_CausalResidualBlock, self).build(
                input_shape
            )  # done to make sure self.built is set True

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer

        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.

        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(
            self.res_output_shape
        )

    def call(self, inputs, training=None):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            training_flag = "training" in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
            self.layers_outputs.append(x)

        if self.final:
            x2 = self.shape_match_conv(inputs)
        else:
            x2 = inputs
        self.layers_outputs.append(x2)
        res_x = layers.add([x2, x])
        self.layers_outputs.append(res_x)

        if self.final_activation is not None:
            res_act_x = self.final_activation(res_x)
            self.layers_outputs.append(res_act_x)
        else:
            res_act_x = res_x
        return res_act_x

    def compute_output_shape(self, input_shape):
        return self.res_output_shape

    def get_config(self):
        config = super(_CausalResidualBlock, self).get_config()
        config.update({"dilation_rate": self.dilation_rate})
        config.update({"nb_filters": self.nb_filters})
        config.update({"kernel_size": self.kernel_size})
        config.update({"kernel_initializer": self.kernel_initializer})
        config.update({"final": self.final})
        return config


class DilatedCnn(BaseEncoder):
    """Create a dilated cnn encoder for model.

    Parameters
    ----------
    optimizer: defaults = None
        Optimizer for algorithm. If not specified Adam is used.
    learning_rate: float, defaults = 0.001
        Learning rate of the algorithm.
    loss: str or Callable, defaults = 'mse'
        Loss function of the algorithm.
    batch_size: int, defaults = 10
        Size of each batch.
    nb_steps: int, defaults = 100
        TODO: no idea what this is

    Note for future self on what I did.
    So first I removed the AutoEncoder and LayerGenerator stuff that just exists
    in this class.
    The params you would pass to the AutoEncoder are just given in the constructor
    in addition to the extra stuff
    _create_encoder and _create_decoder still exist but they use class attributes
    rather than params.
    """

    def __init__(
        self, nb_filters, depth, reduced_size, latent_dim, kernel_size, dropout_rate
    ):
        self.nb_filters = nb_filters
        self.depth = depth
        self.reduced_size = reduced_size
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

    def _compute_adaptive_dilations(self, time_size: int):
        last_dilations = 1
        dilations = [last_dilations]
        rate = 4
        if time_size < 50:
            rate = 2

        while True:
            if last_dilations > time_size / 2:
                break
            last_dilations *= rate
            dilations.append(last_dilations)

        return dilations

    def _create_encoder(self, X: np.ndarray, **kwargs) -> Tuple[Model, Model]:
        dilations = self._compute_adaptive_dilations(X.shape[1])
        dilation_depth = self.depth
        if self.depth > len(dilations):
            dilation_depth = len(dilations)

        i = Input(batch_shape=(None, None, X.shape[2]))
        dilation_size = dilations[0]
        layers_outputs = []

        h = i
        if self.dropout_rate > 0:
            h = Dropout(self.dropout_rate)(h)

        for k in range(dilation_depth):
            h = _CausalResidualBlock(
                nb_filters=self.nb_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_size,
                name="residual_block_{}".format(k),
            )(h)
            layers_outputs.append(h)
            # build newest residual block
            dilation_size = dilations[k]

        # Last layer
        h = _CausalResidualBlock(
            nb_filters=self.reduced_size,
            kernel_size=self.kernel_size,
            dilation_rate=dilation_size,
            name="residual_block_{}".format(dilation_depth),
        )(h)

        h = GlobalMaxPool1D()(h)
        if self.dropout_rate > 0:
            h = Dropout(self.dropout_rate)(h)
        init = initializers.VarianceScaling((1.0 / 3), distribution="uniform")
        h = Dense(self.latent_dim, kernel_initializer=init, bias_initializer=init)(h)
        if X.shape[2]:
            h = layers.Activation("sigmoid")(h)
        layers_outputs.append(h)

        return Model(inputs=i, outputs=h), Model(inputs=i, outputs=layers_outputs)

    def _create_decoder(self, X: np.ndarray, **kwargs) -> Tuple[Model, Model]:
        encoder = self._create_encoder(X)

        dilations = self._compute_adaptive_dilations(X.shape[1])
        dilation_size = dilations[-1]
        dense_size = X.shape[1] * X.shape[2]
        layers_outputs = []
        dilation_depth = self.depth
        if self.depth > len(dilations):
            dilation_depth = len(dilations)
        enc_shape = np.array(K.int_shape(encoder.output)).tolist()
        i = Input(shape=(enc_shape[1],))
        h = Dense(dense_size)(i)
        h = Reshape((X.shape[1], X.shape[2]))(h)
        layers_outputs.append(h)
        for k in range(dilation_depth):
            h = _CausalResidualBlock(
                nb_filters=self.nb_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_size,
                name="decoder_residual_block_{}".format(k),
            )(h)
            layers_outputs.append(h)
            dilation_size = dilations[-(k + 1)]

        h = _CausalResidualBlock(
            nb_filters=X.shape[2],
            kernel_size=self.kernel_size,
            dilation_rate=dilation_size,
            name="residual_block_{}".format(dilation_depth),
        )(h)
        layers_outputs.append(h)

        return Model(inputs=i, outputs=h), Model(inputs=i, outputs=layers_outputs)
