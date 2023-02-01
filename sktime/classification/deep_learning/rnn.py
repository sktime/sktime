#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

"""Time Recurrent Neural Network (RNN) for classification."""

__author__ = ["Markus Löning"]
__all__ = ["SimpleRNNRegressor"]

import tensorflow as tf
from sklearn.utils import check_random_state
from tensorflow import keras

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.rnn import RNNNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class SimpleRNNRegressor(BaseDeepClassifier):
    """Simple recurrent neural network.

    References
    ----------
    ..[1] benchmark forecaster in M4 forecasting competition:
    https://github.com/Mcompetitions/M4-methods
    """

    def __init__(
        self,
        nb_epochs=100,
        batch_size=1,
        units=6,
        callbacks=None,
        random_state=0,
        verbose=0,
        loss="Huber",
        metrics=None,
        activation="sigmoid",
        use_bias=True,
        optimizer=None,
    ):
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.units = units
        self.callbacks = callbacks
        self.random_state = random_state
        self.loss = loss
        self.metrics = metrics
        self.activation = activation
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.history = None
        self._network = RNNNetwork()

        _check_dl_dependencies(severity="error")

        super(SimpleRNNRegressor, self).__init__(
            batch_size=batch_size,
        )

    def build_model(self, input_shape, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training.

        In sktime, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m,d)

        Returns
        -------
        output : a compiled Keras Model
        """
        tf.random.set_seed(self.random_state)
        if self.metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = self.metrics
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        self.optimizer_ = (
            keras.optimizers.Adam(learning_rate=0.01)
            if self.optimizer is None
            else self.optimizer
        )
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=self.loss, optimizer=self.optimizer_, metrics=metrics)
        return model

    def fit(self, X, y, input_checks=True):
        """
        Fit the regressor on the training set (X, y).

        Parameters
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        y : array-like, shape = [n_instances]
            The training data class labels.
        input_checks : boolean
            whether to check the X and y parameters

        Returns
        -------
        self : object
        """
        if self.callbacks is None:
            self._callbacks = []

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.batch_size = int(max(1, min(X.shape[0] / 10, self.batch_size)))

        self.model = self.build_model(self.input_shape)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )
        return self
