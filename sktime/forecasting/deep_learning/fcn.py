# -*- coding: utf-8 -*-
"""FCN for forecasting tem."""

from sktime.networks.fcn import FCNNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class FCNForecaster:
    """Temp docstring."""

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        steps=3,
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
        source = source.transpose(0, 2, 1)
        self.input_shape = source.shape[1:]

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


if __name__ == "__main__":

    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    exog = []
    for i in range(len(raw_seq)):
        exog.append([i, i + 1])
    fcn = FCNForecaster()
    model = fcn._fit(y=raw_seq, X=exog)
