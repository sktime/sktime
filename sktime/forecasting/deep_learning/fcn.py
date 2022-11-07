# -*- coding: utf-8 -*-
"""FCN for forecasting tem."""

from sktime.networks.fcn import FCNNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class FCNForecaster:
    """Temp docstring."""

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
        super(FCNForecaster, self).__init__()
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
            keras.optimizers.Adam(learning_rate=0.001)
            if self.optimizer is None
            else self.optimizer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=self.loss, optimizer=self.optimizer_, metrics=self.metrics)
        return model

    def _predict(self, fh, X=None):
        """Temp docstring."""
        import numpy as np

        currentPred = 1
        lastPred = max(fh)
        fvalues = []
        fh = set(fh)
        source = self.source[-1]
        source = source[np.newaxis, :, :]
        while currentPred <= lastPred:
            yhat = self.model_.predict(source)
            source = np.delete(source, axis=2, obj=0)
            source = np.insert(source, obj=source.shape[-1], values=yhat, axis=-1)
            if currentPred in fh:
                fvalues.append(yhat)

            currentPred += 1
        return fvalues

    def _fit(self, y, fh=None, X=None):
        """Temp docstring."""
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

    def splitSeq(self, steps, seq):
        """Temp."""
        source, target = [], []
        for i in range(len(seq)):
            end_idx = i + steps
            if end_idx > len(seq) - 1:
                break
            seq_src, seq_tgt = seq[i:end_idx], seq[end_idx]
            source.append(seq_src)
            target.append(seq_tgt)
        return source, target
