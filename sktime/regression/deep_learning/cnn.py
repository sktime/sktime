"""Time Convolutional Neural Network (CNN) for regression."""

__author__ = ["AurumnPegasus", "achieveordie"]
__all__ = ["CNNRegressor"]

from copy import deepcopy

from sklearn.utils import check_random_state

from sktime.networks.cnn import CNNNetwork
from sktime.regression.deep_learning.base import BaseDeepRegressor
from sktime.utils.validation._dependencies import _check_dl_dependencies


class CNNRegressor(BaseDeepRegressor):
    """Time Series Convolutional Neural Network (CNN), as described in [1].

    Parameters
    ----------
    should inherited fields be listed here?
    n_epochs       : int, default = 2000
        the number of epochs to train the model
    batch_size      : int, default = 16
        the number of samples per gradient update.
    kernel_size     : int, default = 7
        the length of the 1D convolution window
    avg_pool_size   : int, default = 3
        size of the average pooling windows
    n_conv_layers   : int, default = 2
        the number of convolutional plus average pooling layers
    filter_sizes    : array of shape (n_conv_layers) default = [6, 12]
    random_state    : int or None, default=None
        Seed for random number generation.
    verbose         : boolean, default = False
        whether to output extra information
    loss            : string, default="mean_squared_error"
        fit parameter for the keras model
    activation      : keras.activations or string, default `linear`
        function to use in the output layer.
    optimizer       : keras.optimizers or string, default `None`.
        when `None`, internally uses `keras.optimizers.Adam(0.01)`
    use_bias        : bool, default=True
        whether to use bias in the output layer.
    metrics         : list of strings, default=["accuracy"],

    References
    ----------
    .. [1] Zhao et. al, Convolutional neural networks for
    time series classification, Journal of
    Systems Engineering and Electronics, 28(1):2017.

    Notes
    -----
    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py
    """

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        callbacks=None,
        verbose=False,
        loss="mean_squared_error",
        metrics=None,
        random_state=0,
        activation="linear",
        use_bias=True,
        optimizer=None,
    ):
        _check_dl_dependencies(severity="error")
        super().__init__(
            batch_size=batch_size,
        )
        self.n_conv_layers = n_conv_layers
        self.avg_pool_size = avg_pool_size
        self.kernel_size = kernel_size
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
        self._network = CNNNetwork(
            kernel_size=self.kernel_size,
            avg_pool_size=self.avg_pool_size,
            n_conv_layers=self.n_conv_layers,
            activation=self.activation,
            random_state=self.random_state,
        )

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

        output_layer = keras.layers.Dense(
            units=1,
            activation=self.activation,
            use_bias=self.use_bias,
        )(output_layer)

        self.optimizer_ = (
            keras.optimizers.Adam(learning_rate=0.01)
            if self.optimizer is None
            else self.optimizer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

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
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.model_ = self.build_model(self.input_shape)
        if self.verbose:
            self.model.summary()

        self.history = self.model_.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=deepcopy(self.callbacks) if self.callbacks else [],
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
        from sktime.utils.validation._dependencies import _check_soft_dependencies

        param1 = {
            "n_epochs": 10,
            "batch_size": 4,
            "avg_pool_size": 4,
        }

        param2 = {
            "n_epochs": 12,
            "batch_size": 6,
            "kernel_size": 2,
            "n_conv_layers": 1,
        }
        test_params = [param1, param2]

        if _check_soft_dependencies("keras", severity="none"):
            from keras.callbacks import LambdaCallback

            test_params.append(
                {
                    "n_epochs": 2,
                    "callbacks": [LambdaCallback()],
                }
            )

        return test_params
