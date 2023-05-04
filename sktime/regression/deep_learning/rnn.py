#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

"""Time Recurrent Neural Network (RNN) for regression."""

__author__ = ["mloning"]
__all__ = ["SimpleRNNRegressor"]

from sklearn.utils import check_random_state

from sktime.networks.rnn import RNNNetwork
from sktime.regression.deep_learning.base import BaseDeepRegressor
from sktime.utils.validation._dependencies import _check_dl_dependencies


class SimpleRNNRegressor(BaseDeepRegressor):
    """
    Simple recurrent neural network.

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
        activation="linear",
        use_bias=True,
        optimizer=None,
    ):
        _check_dl_dependencies(severity="error")
        super(SimpleRNNRegressor, self).__init__()
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
        self._network = RNNNetwork(random_state=random_state, units=units)

    def build_model(self, input_shape, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m,d)

        Returns
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf
        from tensorflow import keras

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

    def _fit(self, X, y):
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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                    should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {
            "nb_epochs": 50,
            "batch_size": 2,
            "units": 5,
            "use_bias": False,
        }
        return [params1, params2]
