#!/usr/bin/env python3 -u
"""Time Recurrent Neural Network (RNN) for classification."""

__author__ = ["mloning"]
__all__ = ["SimpleRNNClassifier"]

from copy import deepcopy

from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.rnn import RNNNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies


class SimpleRNNClassifier(BaseDeepClassifier):
    """Simple recurrent neural network.

    Parameters
    ----------
    n_epochs : int, default = 100
        the number of epochs to train the model
    batch_size : int, default = 1
        the number of samples per gradient update.
    units : int, default = 6
        number of units in the network
    callbacks : list of tf.keras.callbacks.Callback objects, default = None
    add_default_callback : bool, default = True
        whether to add default callback
    random_state : int or None, default=0
        Seed for random number generation.
    verbose : boolean, default = False
        whether to output extra information
    loss : string, default="mean_squared_error"
        fit parameter for the keras model
    metrics : list of strings, default=["accuracy"]
        metrics to use in fitting the neural network
    activation : string or a tf callable, default="sigmoid"
        Activation function used in the output layer.
        List of available activation functions: https://keras.io/api/layers/activations/
    use_bias : boolean, default = True
        whether the layer uses a bias vector.
    optimizer : keras.optimizers object, default = RMSprop(lr=0.001)
        specify the optimizer and the learning rate to be used.

    References
    ----------
    ..[1] benchmark forecaster in M4 forecasting competition:
    https://github.com/Mcompetitions/M4-methods
    """

    def __init__(
        self,
        num_epochs=100,
        batch_size=1,
        units=6,
        callbacks=None,
        add_default_callback=True,
        random_state=0,
        verbose=False,
        loss="mean_squared_error",
        metrics=None,
        activation="sigmoid",
        use_bias=True,
        optimizer=None,
    ):
        _check_dl_dependencies(severity="error")
        super().__init__()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.units = units
        self.callbacks = callbacks
        self.add_default_callback = add_default_callback
        self.random_state = random_state
        self.loss = loss
        self.metrics = metrics
        self.activation = activation
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.history = None
        self._network = RNNNetwork(random_state=random_state, units=units)

    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In sktime, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m,d)
        n_classes: int
            The number of classes, which becomes the size of the output layer

        Returns
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        metrics = self.metrics if self.metrics is not None else ["accuracy"]
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)
        output_layer = keras.layers.Dense(
            units=n_classes, activation=self.activation, use_bias=self.use_bias
        )(output_layer)

        self.optimizer_ = (
            keras.optimizers.RMSprop(lr=0.001)
            if self.optimizer is None
            else self.optimizer
        )
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=self.loss, optimizer=self.optimizer_, metrics=metrics)
        return model

    def _fit(self, X, y):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        y : array-like, shape = [n_instances]
            The training data class labels.

        Returns
        -------
        self : object
        """
        from tensorflow import keras

        y_onehot = self.convert_y_to_keras(y)
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]

        self.model_ = self.build_model(self.input_shape, self.n_classes_)

        if self.verbose:
            self.model_.summary()

        # add a ReduceLROnPlateau callback is default is enabled
        # if an instance of ReduceLROnPlateau is already present
        # then don't add it again.
        if self.add_default_callback:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=50,
                min_lr=0.0001,
            )
            if self.callbacks is None:
                self.callbacks_ = [
                    reduce_lr,
                ]
            elif isinstance(self.callbacks, keras.callbacks.Callback):
                self.callbacks_ = [
                    self.callbacks,
                    reduce_lr,
                ]
            elif isinstance(self.callbacks, tuple):
                self.callbacks_ = deepcopy([i for i in self.callbacks])
                if not any(
                    isinstance(callback, keras.callbacks.ReduceLROnPlateau)
                    for callback in self.callbacks
                ):
                    self.callbacks_.append(reduce_lr)
            else:
                raise TypeError(
                    "`callback` can either be None, an instance "
                    "of keras.callbacks.Callback or a tuple containing "
                    "keras.callbacks.Callback objects. "
                    f"But found {type(self.callbacks)} instead."
                )
        else:
            self.callbacks_ = deepcopy(self.callbacks)

        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks_,
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
            "num_epochs": 50,
            "batch_size": 2,
            "units": 5,
            "use_bias": False,
        }
        return [params1, params2]
