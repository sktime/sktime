# -*- coding: utf-8 -*-
"""LongShort Term Memory Fully Convolutional Network (LSTM-FCN)."""
__author__ = ["jnrusson1", "solen0id"]

__all__ = ["LSTMFCNClassifier"]

from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.lstmfcn import LSTMFCNNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class LSTMFCNClassifier(BaseDeepClassifier):
    """

    Implementation of LSTMFCNClassifier from Karim et al (2019) [1].

    Overview
    --------
     Combines an LSTM arm with a CNN arm. Optionally uses an attention mechanism in the
     LSTM which the author indicates provides improved performance.


    Parameters
    ----------
    n_epochs: int, default=2000
     the number of epochs to train the model
    batch_size: int, default=128
        the number of samples per gradient update.
    dropout: float, default=0.8
        controls dropout rate of LSTM layer
    kernel_sizes: list of ints, default=[8, 5, 3]
        specifying the length of the 1D convolution windows
    filter_sizes: int, list of ints, default=[128, 256, 128]
        size of filter for each conv layer
    lstm_size: int, default=8
        output dimension for LSTM layer
    attention: boolean, default=False
        If True, uses custom attention LSTM layer
    callbacks: keras callbacks, default=ReduceLRonPlateau
        Keras callbacks to use such as learning rate reduction or saving best model
        based on validation error
    verbose: 'auto', 0, 1, or 2. Verbosity mode.
        0 = silent, 1 = progress bar, 2 = one line per epoch.
        'auto' defaults to 1 for most cases, but 2 when used with
        `ParameterServerStrategy`. Note that the progress bar is not
        particularly useful when logged to a file, so verbose=2 is
        recommended when not running interactively (eg, in a production
        environment).
    random_state : int or None, default=None
        Seed for random, integer.


    Notes
    -----
    Ported from sktime-dl source code
    https://github.com/sktime/sktime-dl/blob/master/sktime_dl/classification/_lstmfcn.py

    References
    ----------
    .. [1] Karim et al. Multivariate LSTM-FCNs for Time Series Classification, 2019
    https://arxiv.org/pdf/1801.04503.pdf

    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        n_epochs=100,
        batch_size=128,
        dropout=0.8,
        kernel_sizes=(8, 5, 3),
        filter_sizes=(128, 256, 128),
        lstm_size=8,
        attention=False,
        callbacks=None,
        random_state=None,
        verbose=0,
    ):

        super(LSTMFCNClassifier, self).__init__()

        # calced in fit
        self.classes_ = None
        self.input_shape = None
        self.model_ = None
        self.history = None

        # predefined
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.kernel_sizes = kernel_sizes
        self.filter_sizes = filter_sizes
        self.lstm_size = lstm_size
        self.dropout = dropout
        self.attention = attention

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose

        self._network = LSTMFCNNetwork()
        self._is_fitted = False

    def build_model(self, input_shape, n_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training.

        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        n_classes: int
            The number of classes, which shall become the size of the output
             layer
        Returns
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        input_layers, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(units=n_classes, activation="softmax")(
            output_layer
        )

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        if self.callbacks is None:
            self._callbacks = []

        return model

    def _fit(self, X, y):
        """
        Fit the classifier on the training set (X, y).

        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        y : array-like, shape = [n_instances]
            The training data class labels.
        input_checks : boolean
            whether to check the X and y parameters
        validation_X : a nested pd.Dataframe, or array-like of shape =
        (n_instances, series_length, n_dimensions)
            The validation samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
            Unless strictly defined by the user via callbacks (such as
            EarlyStopping), the presence or state of the validation
            data does not alter training in any way. Predictions at each epoch
            are stored in the model's fit history.
        validation_y : array-like, shape = [n_instances]
            The validation class labels.

        Returns
        -------
        self : object
        """
        check_random_state(self.random_state)

        y_onehot = self.convert_y_to_keras(y)

        # Remove?
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]

        self.model_ = self.build_model(self.input_shape, self.n_classes_)

        if self.verbose:
            self.model_.summary()

        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self._callbacks,
        )

        self._is_fitted = True

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
        param1 = {
            "n_epochs": 50,
        }

        param2 = {
            "n_epochs": 100,
        }

        return [param1, param2]
