# -*- coding: utf-8 -*-
import os
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from sktime.clustering._base import TimeSeriesInstances
from sktime.datatypes import convert_to


# Implement loss functions
def mse():
    pass


class BaseEncoder(ABC):
    def _check_X(self, X: TimeSeriesInstances) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = convert_to(X, "numpy3D")
        X = np.reshape(X, (X.shape[0], -1, 1))
        return X

    def create_encoder(self, X: TimeSeriesInstances, **kwargs) -> Tuple[Model, Model]:
        _X = self._check_X(X)
        return self._create_encoder(_X, **kwargs)

    def create_decoder(self, X: TimeSeriesInstances, **kwargs) -> Tuple[Model, Model]:
        _X = self._check_X(X)
        return self._create_encoder(_X, **kwargs)

    @abstractmethod
    def _create_encoder(self, X: np.ndarray, **kwargs) -> Tuple[Model, Model]:
        ...

    @abstractmethod
    def _create_decoder(self, X: np.ndarray, **kwargs) -> Tuple[Model, Model]:
        ...


class AutoEncoder:

    _loss_algo = {"mse": mse}

    def __init__(
        self,
        encoder_model: BaseEncoder,
        X: TimeSeriesInstances,
        optimizer=None,
        learning_rate: float = 0.001,
        loss: str = "mse",
        batch_size: int = 10,
        nb_steps: int = 1000,
    ):
        self.encoder_model = encoder_model
        self.encoder = encoder_model.create_encoder(X)
        self.decoder = encoder_model.create_decoder(X, **{"encoder": self.encoder})
        self.use_vae = False

        self.optimize = optimizer
        self.loss = loss
        self.batch_size = (batch_size,)
        self.nb_steps = nb_steps

        self._optimize = optimizer
        if optimizer is None:
            self._optimize = tf.keras.optimizers.Adam(learning_rate)

        self._loss = loss
        if isinstance(loss, str):
            self._loss = AutoEncoder._loss_algo[loss]

    def __call__(self, inputs, *args, **kwargs):
        return self.decoder(self.encoder(inputs))

    def decoder_predict(self, inputs=None, encoding=None):
        assert inputs is not None or encoding is not None
        if encoding is not None:
            return self.decoder.predict(encoding)
        encoded = self.encoder.predict(inputs)
        if self.use_vae:
            mean, logvar = tf.split(encoded, num_or_size_splits=2, axis=1)
            epsilon = K.random_normal(shape=mean.shape)
            encoded = mean + K.exp(logvar / 2) * epsilon
        return self.decoder.predict(encoded)

    def get_trainable_variables(self):
        trainable_variables = self.encoder.trainable_variables
        if self.decoder is not None:
            trainable_variables += self.decoder.trainable_variables
        return trainable_variables

    def save_weights(self, paths):
        self.encoder.save_weights(paths + "_encoder.h5")
        if self.decoder is not None:
            self.decoder.save_weights(paths + "_decoder.h5")

    def load_weights(self, paths):
        self.encoder.load_weights(paths + "_encoder.h5")
        if self.decoder is not None:
            self.decoder.load_weights(paths + "_decoder.h5")

    def exists(self, paths):
        decoder_exists = True
        if self.decoder is not None:
            decoder_exists = os.path.exists(paths + "_decoder.h5")
        encoder_exists = os.path.exists(paths + "_encoder.h5")
        return decoder_exists and encoder_exists

    def summary(self):
        self.encoder.summary()
        if self.decoder is not None:
            self.decoder.summary()

    def compute_loss(self, batch, noisy_batch=None, training=True):
        return self._loss(batch, noisy_batch, training)
