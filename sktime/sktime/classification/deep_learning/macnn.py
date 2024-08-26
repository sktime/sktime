"""Multi-scale Attention Convolutional Neural Classifier."""

__author__ = ["jnrusson1"]

from copy import deepcopy

from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.macnn import MACNNNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class MACNNClassifier(BaseDeepClassifier):
    """Multi-Scale Attention Convolutional Neural Classifier, as described in [1]_.

    Parameters
    ----------
    n_epochs : int, optional (default=1500)
        The number of epochs to train the model.
    batch_size : int, optional (default=4)
        The number of sample per gradient update.
    padding : str, optional (default="same")
        The type of padding to be provided in MACNN Blocks. Accepts
        all the string values that keras.layers supports.
    pool_size : int, optional (default=3)
        A single value representing pooling windows which are applied
        between two MACNN Blocks.
    strides : int, optional (default=2)
        A single value representing strides to be taken during the
        pooling operation.
    repeats : int, optional (default=2)
        The number of MACNN Blocks to be stacked.
    filter_sizes : tuple, optional (default=(64, 128, 256))
        The input size of Conv1D layers within each MACNN Block.
    kernel_size : tuple, optional (default=(3, 6, 12))
        The output size of Conv1D layers within each MACNN Block.
    reduction : int, optional (default = 16)
        The factor by which the first dense layer of a MACNN Block will be divided by.
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
        then ``keras.optimizers.Adam(learning_rate=0.0001)`` is used.
    callbacks : None or list of keras.callbacks.Callback, optional (default=None)
        The callback(s) to use during training.
    random_state : int, optional (default=0)
        The seed to any random action.
    verbose : bool, optional (default=False)
        Verbosity during model training, making it ``True`` will
        print model summary, training information etc.

    References
    ----------
    .. [1] Wei Chen et. al, Multi-scale Attention Convolutional
    Neural Network for time series classification,
    Neural Networks, Volume 136, 2021, Pages 126-140, ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2021.01.001.

    Examples
    --------
    >>> from sktime.classification.deep_learning.macnn import MACNNClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> macnn = MACNNClassifier(n_epochs=3) # doctest: +SKIP
    >>> macnn.fit(X_train, y_train) # doctest: +SKIP
    MACNNClassifier(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["jnrusson1"],
        "maintainers": "jnrusson1",
        "python_dependencies": "tensorflow",
        # estimator type handled by parent class
    }

    def __init__(
        self,
        n_epochs=1500,
        batch_size=4,
        padding="same",
        pool_size=3,
        strides=2,
        repeats=2,
        filter_sizes=(64, 128, 256),
        kernel_size=(3, 6, 12),
        reduction=16,
        loss="categorical_crossentropy",
        activation="sigmoid",
        use_bias=True,
        metrics=None,
        optimizer=None,
        callbacks=None,
        random_state=0,
        verbose=False,
    ):
        _check_dl_dependencies(severity="error")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides
        self.repeats = repeats
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.loss = loss
        self.activation = activation
        self.use_bias = use_bias
        self.metrics = metrics
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose

        super().__init__()

        self.history = None
        self._network = MACNNNetwork(
            padding=self.padding,
            pool_size=self.pool_size,
            strides=self.strides,
            repeats=self.repeats,
            filter_sizes=self.filter_sizes,
            kernel_size=self.kernel_size,
            reduction=self.reduction,
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

        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=n_classes, activation=self.activation, use_bias=self.use_bias
        )(output_layer)

        self.optimizer_ = (
            keras.optimizers.Adam(learning_rate=0.0001)
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
        y_onehot = self._convert_y_to_keras(y)
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.model_ = self.build_model(self.input_shape, self.n_classes_)
        self.callbacks_ = deepcopy(self.callbacks)

        if self.verbose:
            self.model_.summary()

        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks_,
        )

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, optional (default="default")
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        params1 = {
            "n_epochs": 5,
            "batch_size": 3,
            "filter_sizes": (2, 4, 8),
            "repeats": 1,
        }

        params2 = {
            "n_epochs": 1,
            "filter_sizes": (1, 2, 4),
            "reduction": 8,
            "repeats": 1,
            "random_state": 1,
        }

        return [params1, params2]
