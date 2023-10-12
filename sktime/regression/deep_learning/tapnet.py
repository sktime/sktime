"""Time Convolutional Neural Network (CNN) for classification."""

__author__ = [
    "Jack Russon",
]
__all__ = [
    "TapNetRegressor",
]

from copy import deepcopy

from sklearn.utils import check_random_state

from sktime.networks.tapnet import TapNetNetwork
from sktime.regression.deep_learning.base import BaseDeepRegressor
from sktime.utils.validation._dependencies import _check_dl_dependencies


class TapNetRegressor(BaseDeepRegressor):
    """Time series attentional prototype network (TapNet), as described in [1].

     TapNet was initially proposed for multivariate time series
     classification. The is an adaptation for time series regression. TapNet comprises
     these components: random dimension permutation, multivariate time series
     encoding, and attentional prototype learning.

    Parameters
    ----------
    filter_sizes        : array of int, default = (256, 256, 128)
        sets the kernel size argument for each convolutional block.
        Controls number of convolutional filters
        and number of neurons in attention dense layers.
    kernel_size        : array of int, default = (8, 5, 3)
        controls the size of the convolutional kernels
    layers              : array of int, default = (500, 300)
        size of dense layers
    n_epochs            : int, default = 2000
        number of epochs to train the model
    batch_size          : int, default = 16
        number of samples per update
    dropout             : float, default = 0.5
        dropout rate, in the range [0, 1)
    dilation            : int, default = 1
        dilation value
    activation          : str, default = "sigmoid"
        activation function for the last output layer
    loss                : str, default = "mean_squared_error"
        loss function for the classifier
    optimizer           : str or None, default = "Adam(lr=0.01)"
        gradient updating function for the classifier
    use_bias            : bool, default = True
        whether to use bias in the output dense layer
    use_rp              : bool, default = True
        whether to use random projections
    use_att             : bool, default = True
        whether to use self attention
    use_lstm        : bool, default = True
        whether to use an LSTM layer
    use_cnn         : bool, default = True
        whether to use a CNN layer
    verbose         : bool, default = False
        whether to output extra information
    random_state    : int or None, default = None
        seed for random

    References
    ----------
    .. [1] Zhang et al. Tapnet: Multivariate time series classification with
    attentional prototypical network,
    Proceedings of the AAAI Conference on Artificial Intelligence
    34(4), 6845-6852, 2020

    Notes
    -----
    The Implementation of TapNet found at https://github.com/kdd2019-tapnet/tapnet
    Currently does not implement custom distance matrix loss function
    or class  based self attention.
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        dropout=0.5,
        filter_sizes=(256, 256, 128),
        kernel_size=(8, 5, 3),
        dilation=1,
        layers=(500, 300),
        use_rp=True,
        activation=None,
        rp_params=(-1, 3),
        use_bias=True,
        use_att=True,
        use_lstm=True,
        use_cnn=True,
        random_state=None,
        padding="same",
        loss="mean_squared_error",
        optimizer=None,
        metrics=None,
        callbacks=None,
        verbose=False,
    ):
        _check_dl_dependencies(severity="error")
        super().__init__()

        self.batch_size = batch_size
        self.random_state = random_state
        self.kernel_size = kernel_size
        self.layers = layers
        self.rp_params = rp_params
        self.filter_sizes = filter_sizes
        self.activation = activation
        self.use_att = use_att
        self.use_bias = use_bias

        self.dilation = dilation
        self.padding = padding
        self.n_epochs = n_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.callbacks = callbacks
        self.verbose = verbose

        self.dropout = dropout
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # parameters for random projection
        self.use_rp = use_rp
        self.rp_params = rp_params

        self._network = TapNetNetwork(
            dropout=self.dropout,
            filter_sizes=self.filter_sizes,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            layers=self.layers,
            use_rp=self.use_rp,
            rp_params=self.rp_params,
            use_att=self.use_att,
            use_lstm=self.use_lstm,
            use_cnn=self.use_cnn,
            random_state=self.random_state,
            padding=self.padding,
        )

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

        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=1, activation=self.activation, use_bias=self.use_bias
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
        # Transpose to conform to expectation format from keras
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]

        self.model_ = self.build_model(self.input_shape)
        if self.verbose:
            self.model_.summary()
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
            "padding": "valid",
            "filter_sizes": (16, 16, 16),
            "kernel_size": (3, 3, 1),
            "layers": (25, 50),
        }
        param2 = {
            "n_epochs": 20,
            "use_cnn": False,
            "layers": (25, 25),
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
