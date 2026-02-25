"""Time Le-Net (TLENET)."""

__author__ = ["JamesLarge, Withington"]
__all__ = ["TleNetClassifier"]

from copy import deepcopy

from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.tlenet import TleNetNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class TleNetClassifier(BaseDeepClassifier):
    """Time Le-Net (TLENET) from Fawaz et. al.

    Parameters
    ----------
    n_epochs: int, default=2000
     the number of epochs to train the model
    batch_size: int, default=128
        the number of samples per gradient update.
    warping_ratio: list of float, default=[0.5, 1, 2]
        warping ratio for each window
    slice_ratio: float, default=0.1
        ratio of the time series used to create a slice
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

    Attributes
    ----------
    n_classes: int
        the number of classes

    References
    ----------
    .. [1] Fawaz et. al. Data augmentation for time series
    classification using convolutional neural networks

    Notes
    -----
    This implementation is based on the
        https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/tlenet.py
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["jamesl", "withington"],
        "maintainers": ["jamesl", "withington", "nilesh05apr"],
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        n_epochs=2000,
        batch_size=128,
        warping_ratio=None,
        slice_ratio=0.1,
        callbacks="ReduceLRonPlateau",
        verbose="auto",
        random_state=None,
        loss="categorical_crossentropy",
        optimizer=None,
        metrics=None,
    ):
        _check_dl_dependencies(severity="error")
        super().__init__()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.warping_ratio = warping_ratio
        self.slice_ratio = slice_ratio
        self.callbacks = callbacks
        self.verbose = verbose
        self.random_state = random_state
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        self._network = TleNetNetwork(
            warping_ratio=self.warping_ratio,
            slice_ratio=self.slice_ratio,
            random_state=self.random_state,
        )

    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a complied, un-trained, keras model that is ready for training.

        In sktime, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape     : tuple
            The shape of the data fed into the input layer, should be (m, d)
        n_classes       : int
            The number of classes, which becomes the size of the output layer

        Returns
        -------
        output: a compiled Keras model
        """
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        if self.warping_ratio is None:
            self.warping_ratio = [0.5, 1, 2]

        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(n_classes, activation="softmax")(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        # if user hasn't provided own metrics use accuracy
        if self.metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = self.metrics

        model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.Adam(),
            metrics=metrics,
        )

        return model

    def _fit(self, X, y, **kwargs):
        """Fit the model to the training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The training input samples.
        y : array-like, shape = (n_samples,)
            The target values.
        """
        y_onehot = self.convert_y_to_keras(y)
        X, y_onehot, _ = self._network.pre_processing(X, y_onehot, None)
        # Transpose to conform to expectation format by keras
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.n_classes_ = y_onehot.shape[1]

        self.model_ = self.build_model(self.input_shape, self.n_classes_)
        if self.verbose:
            self.model_.summary()
        self.history = self.model_.fit(
            X,
            y_onehot,
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
            "n_epochs": 25,
            "batch_size": 4,
            "warping_ratio": [0.5, 1, 2],
            "slice_ratio": 0.1,
        }

        param2 = {
            "n_epochs": 5,
            "batch_size": 2,
            "warping_ratio": [0.25, 0.5, 0.75],
            "slice_ratio": 0.3,
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
