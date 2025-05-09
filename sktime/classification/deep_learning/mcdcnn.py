"""Multi Channel Deep Convolutional Neural Classifier (MCDCNN)."""

__author__ = ["James-Large"]

from copy import deepcopy

import numpy as np
from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.mcdcnn import MCDCNNNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class MCDCNNClassifier(BaseDeepClassifier):
    """Multi Channel Deep Convolutional Neural Classifier, as described in [1]_.

    Adapted from the implementation of Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mcdcnn.py

    Parameters
    ----------
    n_epochs : int, optional (default=120)
        The number of epochs to train the model.
    batch_size : int, optional (default=16)
        The number of samples per gradient update.
    kernel_size : int, optional (default=5)
        The size of kernel in Conv1D layer.
    pool_size : int, optional (default=2)
        The size of kernel in (Max) Pool layer.
    filter_sizes : tuple, optional (default=(8, 8))
        The sizes of filter for Conv1D layer corresponding
        to each Conv1D in the block.
    dense_units : int, optional (default=732)
        The number of output units of the final Dense
        layer of this Network. This is NOT the final layer
        but the penultimate layer.
    conv_padding : str or None, optional (default="same")
        The type of padding to be applied to convolutional
        layers.
    pool_padding : str or None, optional (default="same")
        The type of padding to be applied to pooling layers.
    loss : str, optional (default="categorical_crossentropy")
        The name of the loss function to be used during training,
        should be supported by keras.
    activation : str, optional (default="sigmoid")
        The activation function to apply at the output. It should be
        "software" if response variable has more than two types.
    use_bias : bool, optional (default=True)
        Whether bias should be included in the output layer.
    metrics : None or string, optional (default=None)
        The string which will be used during model compilation. If left as None,
        then "accuracy" is passed to ``model.compile()``.
    optimizer: None or keras.optimizers.Optimizer instance, optional (default=None)
        The optimizer that is used for model compiltation. If left as None,
        then ``keras.optimizers.SGD`` is used with the following parameters -
        ``learning_rate=0.01, momentum=0.9, weight_decay=0.0005``.
    callbacks : None or list of keras.callbacks.Callback, optional (default=None)
        The callback(s) to use during training.
    random_state : int, optional (default=0)
        The seed to any random action.

    References
    ----------
    .. [1] Zheng et. al, Time series classification using multi-channels deep
      convolutional neural networks, International Conference on
      Web-Age Information Management, Pages 298-310, year 2014, organization: Springer.

    Examples
    --------
    >>> from sktime.classification.deep_learning.mcdcnn import MCDCNNClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_tain = load_unit_test(split="train")
    >>> mcdcnn = MCDCNNClassifier()     # doctest: +SKIP
    >>> mcdcnn.fit(X_train, y_train)    # doctest: +SKIP
    MCDCNNClassifier(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["hfawaz", "james-large"],
        "maintainers": ["james-large"],
        "python_dependencies": "tensorflow",
        # estimator type handled by parent class
    }

    def __init__(
        self,
        n_epochs=120,
        batch_size=16,
        kernel_size=5,
        pool_size=2,
        filter_sizes=(8, 8),
        dense_units=732,
        conv_padding="same",
        pool_padding="same",
        loss="categorical_crossentropy",
        activation="sigmoid",
        use_bias=True,
        callbacks=None,
        metrics=None,
        optimizer=None,
        verbose=False,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.filter_sizes = filter_sizes
        self.dense_units = dense_units
        self.conv_padding = conv_padding
        self.pool_padding = pool_padding
        self.loss = loss
        self.activation = activation
        self.use_bias = use_bias
        self.callbacks = callbacks
        self.metrics = metrics
        self.optimizer = optimizer
        self.verbose = verbose
        self.random_state = random_state

        super().__init__()

        self.history = None
        self._network = MCDCNNNetwork(
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            filter_sizes=self.filter_sizes,
            dense_units=self.dense_units,
            conv_padding=self.conv_padding,
            pool_padding=self.pool_padding,
            random_state=self.random_state,
        )

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

        metrics = ["accuracy"] if self.metrics is None else self.metrics

        input_layers, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=n_classes,
            activation=self.activation,
            use_bias=self.use_bias,
        )(output_layer)

        self.optimizer_ = (
            keras.optimizers.SGD(
                learning_rate=0.01,
                momentum=0.9,
                weight_decay=0.0005,
            )
            if self.optimizer is None
            else self.optimizer
        )

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )

        return model

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
        y_onehot = self._convert_y_to_keras(y)
        X = X.transpose(0, 2, 1)
        self.input_shape = X.shape[1:]
        X = self._network._prepare_input(X)

        check_random_state(self.random_state)

        self.model_ = self.build_model(self.input_shape, self.n_classes_)
        self.callbacks_ = deepcopy(self.callbacks)

        if self.verbose:
            self.model_.summary()

        self.history = self.model_.fit(
            X,
            y_onehot,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=self.callbacks_,
        )

        return self

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
        self.check_is_fitted()
        X = X.transpose([0, 2, 1])
        X = self._network._prepare_input(X)

        probs = self.model_.predict(X, self.batch_size, **kwargs)

        if probs.shape[1] == 1:
            probs = np.hstack([1 - probs, probs])
        probs = probs / probs.sum(axis=1, keepdims=1)
        return probs
