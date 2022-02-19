# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import numpy as np

from sktime.clustering._base import BaseClusterer, TimeSeriesInstances


class BaseDnnClusterer(BaseClusterer, ABC):
    def __init__(
        self, encoder_model: str, n_clusters: str, optimizer: str, verbose: bool = False
    ):
        self.encoder_model = encoder_model
        self.n_clusters = n_clusters
        self.optimizer = optimizer
        self.verbose = verbose

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        pass

    def _fit(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        pass

    # def pretrain_encoder(self, x, nb_steps, log_writer, y=None, x_test=None, y_test=None,
    #                      verbose=True, noise=None):
    #     if verbose:
    #         self.encoder_model.summary()
    #
    #     self.last_logs_enc = {'ari': 0.0, 'nmi': 0.0, 'acc': 0.0,
    #                           'max_ari': -np.inf, 'max_nmi': -np.inf, 'max_acc': -np.inf}
    #
    #     i = 0  # Number of performed optimization steps
    #     epochs = 0  # Number of performed epochs
    #
    #     train_loss = tf.keras.metrics.Mean(name='train_loss')
    #
    #     # define the train function
    #     @tf.function
    #     def train_step(x_batch, noisy_batch):
    #         with tf.GradientTape() as tape:
    #             loss = self.encoder_model.loss.compute_loss(x_batch, noisy_batch=noisy_batch)
    #         gradients = tape.gradient(loss, self.encoder_model.get_trainable_variables())
    #         self.optimizer.apply_gradients(zip(gradients, self.encoder_model.get_trainable_variables()))
    #
    #         train_loss(loss)
    #
    #     if verbose:
    #         print('start pre-train')
    #     # Encoder training
    #     while i < nb_steps:
    #         train_loss.reset_states()
    #         train_ds = tf.data.Dataset.from_tensor_slices(x)
    #         train_ds = train_ds.shuffle(x.shape[0], reshuffle_each_iteration=True)
    #         train_ds = train_ds.batch(self.batch_size).as_numpy_iterator()
    #         for batch in train_ds:
    #             if noise is not None:
    #                 noisy_batch = utils.noise(batch, noise)
    #             else:
    #                 noisy_batch = batch
    #             train_step(batch, noisy_batch)
    #
    #             i += 1
    #             if i >= nb_steps:
    #                 break
    #
    #         if verbose:
    #             template = 'Epoch {}, Loss: {}'
    #             print(template.format(epochs + 1, train_loss.result()))
    #         epochs += 1
    #         y_pred = self.log_stats_encoder(x, y, x_test, y_test,
    #                                         [train_loss.result(), 0, train_loss.result()],
    #                                         epochs, log_writer, 'pretrain')
    #
    #     if verbose:
    #         print('end of pre-train')
    #
    #
    # @abstractmethod
    # def pretrain_encoder(self):
