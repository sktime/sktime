"""Contextual Time-series Neural Classifier for TSC."""

__author__ = ["James-Large", "TonyBagnall", "AurumnPegasus"]
__all__ = ["CNTCClassifier"]
from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.cntc import CNTCNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class CNTCClassifier(BaseDeepClassifier):
    """Contextual Time-series Neural Classifier (CNTC), as described in [1].

    Adapted from the implementation from Fullah et. al
    https://github.com/AmaduFullah/CNTC_MODEL/blob/master/cntc.ipynb

    Parameters
    ----------
    n_epochs       : int, default = 2000
        the number of epochs to train the model
    batch_size      : int, default = 16
        the number of samples per gradient update.
    filter_sizes    : tuple of shape (2), default = (16, 8)
        filter sizes for CNNs in CCNN arm.
    kernel_sizes     : two-tuple, default = (1, 1)
        the length of the 1D convolution window for
        CNNs in CCNN arm.
    rnn_size        : int, default = 64
        number of rnn units in the CCNN arm.
    lstm_size       : int, default = 8
        number of lstm units in the CLSTM arm.
    dense_size      : int, default = 64
        dimension of dense layer in CNTC.
    random_state    : int or None, default=None
        Seed for random number generation.
    verbose         : boolean, default = False
        whether to output extra information
    loss            : string, default="mean_squared_error"
        fit parameter for the keras model
    optimizer       : keras.optimizer, default=keras.optimizers.Adam(),
    metrics         : list of strings, default=["accuracy"],

    References
    ----------
    .. [1] Network originally defined in:
        @article{FULLAHKAMARA202057,
        title = {Combining contextual neural networks for time series classification},
        journal = {Neurocomputing},
        volume = {384},
        pages = {57-66},
        year = {2020},
        issn = {0925-2312},
        doi = {https://doi.org/10.1016/j.neucom.2019.10.113},
        url = {https://www.sciencedirect.com/science/article/pii/S0925231219316364},
        author = {Amadu {Fullah Kamara} and Enhong Chen and Qi Liu and Zhen Pan},
        keywords = {Time series classification, Contextual convolutional neural
            networks, Contextual long short-term memory, Attention, Multilayer
            perceptron},
       }

    Examples
    --------
    >>> from sktime.classification.deep_learning.cntc import CNTCClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> cntc = CNTCClassifier() # doctest: +SKIP
    >>> cntc.fit(X_train, y_train) # doctest: +SKIP
    CNTCClassifier(...) # doctest: +SKIP
    """

    _tags = {
        "authors": [
            "AmaduFullah",
            "James-Large",
            "Withington",
            "TonyBagnall",
            "AurumnPegasus",
        ],
        "maintainers": ["James-Large", "Withington", "AurumnPegasus"],
        "python_dependencies": ["tensorflow", "keras-self-attention"],
    }

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        filter_sizes=(16, 8),
        kernel_sizes=(1, 1),
        rnn_size=64,
        lstm_size=8,
        dense_size=64,
        callbacks=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")

        self.kernel_sizes = kernel_sizes  # used plural
        self.filter_sizes = filter_sizes  # used plural
        self.rnn_size = rnn_size
        self.lstm_size = lstm_size
        self.dense_size = dense_size
        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.random_state = random_state

        super().__init__()

        self._network = CNTCNetwork()

    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In sktime, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (d,m)
        n_classes: int
            The number of classes, which becomes the size of the output layer

        Returns
        -------
        output : a compiled Keras Model
        """
        from tensorflow import keras

        metrics = ["accuracy"] if self.metrics is None else self.metrics
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(units=n_classes, activation="softmax")(
            output_layer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.Adam(),
            metrics=metrics,
        )
        return model

    def prepare_input(self, X):
        """
        Prepare input for the CLSTM arm of the model.

        According to the paper:
            "
                Time series data is fed into a CLSTM and CCNN networks simultaneously
                and is perceived differently. In the CLSTM block, the input data is
                viewed as a multivariate time series with a single time stamp. In
                contrast, the CCNN block receives univariate data with numerous time
                stamps
            "

        Arguments
        ---------
        X: tuple of shape = (series_length (m), n_dimensions (d))
            The shape of the data fed into the model.

        Returns
        -------
        trainX: tuple,
            The input to be fed to the two arms of CNTC.
        """
        import numpy as np
        import pandas as pd
        from tensorflow import keras

        if X.shape[2] == 1:
            # Converting data to pandas
            trainX1 = X.reshape([X.shape[0], X.shape[1]])
            pd_trainX = pd.DataFrame(trainX1)

            # Taking rolling window
            window = pd_trainX.rolling(window=3).mean()
            window = window.fillna(0)

            trainX2 = np.concatenate((trainX1, window), axis=1)
            trainX2 = keras.backend.variable(trainX2)
            trainX2 = keras.layers.Dense(
                trainX1.shape[1], input_shape=(trainX2.shape[1:])
            )(trainX2)
            trainX2 = keras.backend.eval(trainX2)
            trainX = trainX2.reshape((trainX2.shape[0], trainX2.shape[1], 1))
        else:
            trainXs = []
            for i in range(X.shape[2]):
                trainX1 = X[:, :, i]
                pd_trainX = pd.DataFrame(trainX1)

                window = pd_trainX.rolling(window=3).mean()
                window = window.fillna(0)

                trainX2 = np.concatenate((trainX1, window), axis=1)
                trainX2 = keras.backend.variable(trainX2)
                trainX2 = keras.layers.Dense(
                    trainX1.shape[1], input_shape=(trainX2.shape[1:])
                )(trainX2)
                trainX2 = keras.backend.eval(trainX2)

                trainX = trainX2.reshape((trainX2.shape[0], trainX2.shape[1], 1))
                trainXs.append(trainX)

            trainX = np.concatenate(trainXs, axis=2)
        return trainX

    def _fit(self, X, y):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape = (n_instances (n), n_dimensions (d), series_length (m))
            The training input samples.
        y : np.ndarray of shape n
            The training data class labels.

        Returns
        -------
        self : object
        """
        if self.callbacks is None:
            self._callbacks = []
        y_onehot = self._convert_y_to_keras(y)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.model_ = self.build_model(self.input_shape, self.n_classes_)
        X2 = self.prepare_input(X)
        if self.verbose:
            self.model_.summary()
        self.history = self.model_.fit(
            [X2, X, X],
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self._callbacks,
        )
        return self

    def _predict(self, X, **kwargs):
        import numpy as np

        probs = self._predict_proba(X, **kwargs)
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in probs
            ]
        )

    def _predict_proba(self, X, **kwargs):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : an np.ndarray of shape = (n_instances, n_dimensions, series_length)
            The training input samples.

        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        import numpy as np

        X2 = self.prepare_input(X)
        probs = self.model_.predict([X2, X, X], self.batch_size, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        probs = probs / probs.sum(axis=1, keepdims=1)
        return probs
