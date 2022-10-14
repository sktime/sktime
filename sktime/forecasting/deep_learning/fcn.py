# -*- coding: utf-8 -*-
"""FCN for forecasting."""
from sktime.networks.fcn import FCNNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class FCNForecaster:
    """Temp docstring."""

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        callbacks=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        random_state=None,
        activation="sigmoid",
        use_bias=True,
        optimizer=None,
    ):
        _check_dl_dependencies(severity="error")
        super(FCNForecaster, self).__init__()
        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.random_state = random_state
        self.activation = activation
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.history = None
        self._network = FCNNetwork()

    def build_model(self, input_shape, **kwargs):
        """Temp docstring."""
        from tensorflow import keras

        self.metrics = ["accuracy"] if self.metrics is None else self.metrics
        input_layer, output_layer = self._network.build_network(
            input_shape,
        )
        output_layer = keras.layers.Dense(units=1, activation="sigmoid")(output_layer)

        self.optimizer_ = (
            keras.optimiqzers.Adam(learning_rate=0.001)
            if self.optimzer is None
            else self.optimizer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=self.loss, optimizer=self.optimizer_, metrics=self.metrics)
        return model

    def _fit(self, X, y):
        """Temp docstring."""
        X = X.transpose(0, 2, 1)
        self.input_shape = X.shape[1:]
        self.model_ = self.build_model(self.input_shape, 1)
        self.model_.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )
        return self
