# -*- coding: utf-8 -*-
"""Multi Layer Perceptron Network (MLP) for forecasting."""

from sktime.forecasting.deep_learning.base import BaseDeepForecastor
from sktime.networks.mlp import MLPNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class MLPForecaster(BaseDeepForecastor):
    """Multi Layer Perceptron Network (MLP), derived from [1].

    Parameters
    ----------
    n_epochs       : int, default = 2000
        the number of epochs to train the model
    batch_size      : int, default = 4
        the number of samples per gradient update.
    steps           : int, default = 3
        the lookback window for forecasting.
    random_state    : int or None, default=None
        Seed for random number generation.
    verbose         : boolean, default = False
        whether to output extra information
    loss            : string, default="mean_squared_error"
        fit parameter for the keras model
    optimizer       : keras.optimizer, default=keras.optimizers.Adam(),
    metrics         : list of strings, default=["accuracy"],
    activation      : string or a tf callable, default="sigmoid"
        Activation function used in the output linear layer.
        List of available activation functions:
        https://keras.io/api/layers/activations/
    use_bias        : boolean, default = True
        whether the layer uses a bias vector.
    optimizer       : keras.optimizers object, default = Adam(lr=0.01)
        specify the optimizer and the learning rate to be used.

    Notes
    -----
    .. .. [1]  Network originally defined in:
    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }

    Derived from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py
    """

    def __init__(
        self,
        n_epochs=200,
        batch_size=4,
        steps=3,
        callbacks=None,
        verbose=False,
        loss="mse",
        metrics=None,
        random_state=None,
        activation="relu",
        use_bias=True,
        optimizer=None,
    ):
        _check_dl_dependencies(severity="error")
        super(MLPForecaster, self).__init__()
        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.steps = steps
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.random_state = random_state
        self.activation = activation
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.history = None
        self._network = MLPNetwork()

    def build_model(self, input_shape, **kwargs):
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
        from tensorflow import keras

        self.metrics = ["accuracy"] if self.metrics is None else self.metrics
        input_layer, output_layer = self._network.build_network(
            input_shape,
        )
        output_layer = keras.layers.Dense(units=1, activation=self.activation)(
            output_layer
        )

        self.optimizer_ = (
            keras.optimizers.Adam(learning_rate=0.01)
            if self.optimizer is None
            else self.optimizer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=self.loss, optimizer=self.optimizer_, metrics=self.metrics)
        return model

    def _fit(self, y, fh=None, X=None):
        """Fit the forecaster on the training set (y) with exog data (X).

        Parameters
        ----------
        y: np.array of shape = (n_instances (n))
            The main data which needs to be predicted.
        fh: list of int
            Forecasting Horizon for the forecaster.
        X: np.ndarray of shape = (n_instances (n), exog_dimensions (d))
            Exogeneous data for data prediction.

        Returns
        -------
        self: object
        """
        import numpy as np

        source, target = self.splitSeq(self.steps, y)
        if X is not None:
            src_x, _ = self.splitSeq(self.steps, X)
            # currently takes care of cases where exog data is
            # greater than 1 in length
            source = [
                [_sx + [_sy] for _sx, _sy in zip(sx, sy)]
                for sx, sy in zip(src_x, source)
            ]

        source, target = np.array(source), np.array(target)
        if X is None:
            source = source.reshape((*source.shape, 1))
        source = source.transpose(0, 2, 1)
        self.input_shape = source.shape[1:]
        self.source, self.target = source, target

        self.model_ = self.build_model(self.input_shape)
        if self.verbose:
            self.model_.summary()

        self.history = self.model_.fit(
            source,
            target,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )
        return self
