# -*- coding: utf-8 -*-
"""Residual Network (ResNet) for Regression."""

__author__ = ["James Large", "Withington", "nilesh05apr"]
__all__ = [
    "ResNetRegressor",
]

from sktime.networks.resnet import ResNetNetwork
from sktime.regression.deep_learning.base import BaseDeepRegressor
from sktime.utils.validation._dependencies import _check_dl_dependencies


class ResNetRegressor(BaseDeepRegressor):
    """Residual Network (ResNet) for Regression, as described in [1].

    Parameters
    ----------
    n_epochs            : int, default = 2000
        number of epochs to train the model
    batch_size          : int, default = 16
        number of samples per update
    random_state        : int, default = None
        seed for random, integer
    verbose             : bool, default = False
        whether to print training progress to stdout
    optimizer           : str or None, default = "Adam(lr=0.01)"
        gradient updating function for the classifier
    loss                : str, default = "mean_squared_error"
        loss function for the classifier
    activation          : str, default = "sigmoid"
        activation function for the last output layer
    use_bias            : bool, default = True
        whether to use bias in the output dense layer
    random_projection   : bool, default = True
        whether to use random projections
    metrics             : list, default = ['mean_squared_error']
        list of metrics to evaluate the model on

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    References
    ----------
        .. [1] Wang et. al, Time series classification from
    scratch with deep neural networks: A strong baseline,
    International joint conference on neural networks (IJCNN), 2017.
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        random_state=None,
        verbose=False,
        optimizer=None,
        metrics="mean_squared_error",
        loss="mean_squared_error",
        activation=None,
        use_bias=True,
        callbacks=None,
    ):
        _check_dl_dependencies(severity="error")
        super(ResNetRegressor, self).__init__()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.activation = activation
        self.use_bias = use_bias
        self.callbacks = callbacks
        self._network = ResNetNetwork()

    def build_model(self, input_shape, **kwargs):
        """Construct a complied, un-trained, keras model that is ready for training.

        In sktime, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape     : tuple
            The shape of the data fed into the input layer, should be (m, d)

        Returns
        -------
        output: a compiled Keras model
        """
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        metrics = ["mean_squared_error"] if self.metrics is None else self.metrics

        input_layer, output_layer = self._network.build_network(
            input_shape=input_shape, **kwargs
        )

        output_layer = keras.layers.Dense(
            units=1, activation=self.activation, use_bias=self.use_bias
        )(output_layer)

        self.optimizer_ = (
            keras.optimizers.Adam(lr=0.01) if self.optimizer is None else self.optimizer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )

        return model

    def _fit(self, X, y):
        """Fit the regressor on the training set (X, y).

        Parameters
        ----------
        X   : np.ndarray of shape = (n_instances(n), n_dimensions(d), series_length(m))
            Input training samples
        y   : np.ndarray of shape n
            Input training responses

        Returns
        -------
        self: object
        """
        if self.callbacks is None:
            self.callbacks = []

        X = X.transpose(0, 2, 1)
        self.input_shape_ = X.shape[1:]

        self.model_ = self.build_model(self.input_shape_)

        if self.verbose:
            self.model_.summary()

        self.history = self.model_.fit(
            X,
            y,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
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
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        params1 = {
            "n_epochs": 15,
            "batch_size": 4,
            "random_state": 1,
        }

        params2 = {
            "n_epochs": 10,
            "batch_size": 8,
        }

        return [params1, params2]
