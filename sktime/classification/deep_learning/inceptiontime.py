"""InceptionTime for classification."""

__all__ = ["InceptionTimeClassifier"]

import warnings
from copy import deepcopy

from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.inceptiontime import InceptionTimeNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class InceptionTimeClassifier(BaseDeepClassifier):
    """InceptionTime Deep Learning Classifier.

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

    Described in [1]_, InceptionTime is a deep learning model designed for
    time series classification. It is based on the Inception architecture
    for images. The model is made up of a series of Inception modules.

    ``InceptionTimeClassifier`` is a single instance of InceptionTime model
    described in the original publication [1]_, which uses an ensemble of 5
    single instances.

    To build an ensemble of models mirroring [1]_, use the ``BaggingClassifier`` with
    ``n_estimators=5``, ``bootstrap=False``, and ``estimator`` being an instance of
    this ``InceptionTimeClassifier``.

    Parameters
    ----------
    activation : string or a tf callable, default="softmax"
        Activation function used in the output layer.
        List of available activation functions:
        https://keras.io/api/layers/activations/
    activation_hidden : string or a tf callable, default="relu"
        Activation function used in the hidden layers.
        List of available activation functions:
        https://keras.io/api/layers/activations/
    activation_inception : string or a tf callable, default="linear"
        Activation function used in the Inception modules.
        List of available activation functions:
        https://keras.io/api/layers/activations/
    n_epochs : int, default=1500
    batch_size : int, default=64
        the number of samples per gradient update
    kernel_size : int, default=40
        specifying the length of the 1D convolution window
    n_filters : int, default=32
    use_residual : boolean, default=True
    use_bottleneck : boolean, default=True
    bottleneck_size : int, default=32
    depth : int, default=6
    callbacks : list of tf.keras.callbacks.Callback objects
    random_state: int, optional, default=None
        random seed for internal random number generator
    verbose: boolean, default=False
        whether to print runtime information
    loss: str, default="categorical_crossentropy"
    metrics: optional
    class_weight: dict, optional, default=None
        Dictionary mapping class labels to a weight (float) value to
        be used during model training.
        For example, ``{"A": 1.0, "B": 2.5}`` will assign a weight of 1.0 to class "A"
        and 2.5 to class "B".
        This is passed directly to Keras' ``fit`` method as the ``class_weight``
        argument after converting labels to integer encoding.
        If None, all classes are given equal weight.

    Notes
    -----
    ..[1] Fawaz et. al, InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    Examples
    --------
    Single instance of InceptionTime model:
    >>> from sktime.classification.deep_learning import InceptionTimeClassifier
    >>> from sktime.datasets import load_unit_test  # doctest: +SKIP
    >>> X_train, y_train = load_unit_test(split="train")  # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test")  # doctest: +SKIP
    >>> clf = InceptionTimeClassifier()  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    InceptionTimeClassifier(...)

    To build an ensemble of models mirroring [1]_, use the ``BaggingClassifier``
    as follows:
    >>> from sktime.classification.ensemble import BaggingClassifier
    >>> from sktime.classification.deep_learning import InceptionTimeClassifier
    >>> from sktime.datasets import load_unit_test  # doctest: +SKIP
    >>> X_train, y_train = load_unit_test(split="train")  # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test")  # doctest: +SKIP
    >>> clf = BaggingClassifier(
    ...     InceptionTimeClassifier(),
    ...     n_estimators=5,
    ...     bootstrap=False
    ... )  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    BaggingClassifier(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["hfawaz", "james-large", "noxthot"],
        "maintainers": ["james-large"],
        # estimator type handled by parent class
        # capabilities
        # ------------
        "capability:class_weight": True,
        # testing configuration
        # ---------------------
        "tests:skip_by_name": ["test_fit_idempotent"],
        "tests:libs": ["sktime.networks.inceptiontime"],
        "tests:vm": True,
    }

    def __init__(
        self,
        n_epochs=1500,
        batch_size=64,
        kernel_size=40,
        n_filters=32,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        class_weight=None,
        activation="softmax",
        activation_hidden="relu",
        activation_inception="linear",
    ):
        _check_dl_dependencies(severity="error")

        # predefined
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.activation_inception = activation_inception
        self.batch_size = batch_size
        self.bottleneck_size = bottleneck_size
        self.callbacks = callbacks
        self.depth = depth
        self.kernel_size = kernel_size
        self.loss = loss
        self.metrics = metrics
        self.n_epochs = n_epochs
        self.n_filters = n_filters
        self.random_state = random_state
        self.use_bottleneck = use_bottleneck
        self.use_residual = use_residual
        self.verbose = verbose
        self.class_weight = class_weight

        super().__init__()

        network_params = {
            "activation": self.activation_hidden,
            "activation_inception": self.activation_inception,
            "n_filters": n_filters,
            "use_residual": use_residual,
            "use_bottleneck": use_bottleneck,
            "bottleneck_size": bottleneck_size,
            "depth": depth,
            "kernel_size": kernel_size,
            "random_state": random_state,
        }

        self._network = InceptionTimeNetwork(**network_params)

    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
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
        from tensorflow import keras

        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            n_classes,
            activation=self.activation,
        )(output_layer)

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
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.model_ = self.build_model(self.input_shape, self.n_classes_)
        if self.verbose:
            self.model_.summary()

        callbacks = self._check_callbacks(self.callbacks)

        # Convert class_weight dict from label to integer encoding
        class_weight = self.class_weight
        if class_weight is not None:
            valid_labels = set(self.label_encoder.classes_)
            # keep only labels present in training data
            filtered_class_weight = {
                self.label_encoder.transform([label])[0]: weight
                for label, weight in class_weight.items()
                if label in valid_labels
            }
            if len(filtered_class_weight) < len(class_weight):
                warnings.warn(
                    "class_weight contains labels not observed in the training data; "
                    "these labels are ignored.",
                    UserWarning,
                )
            # if nothing valid left, set to None so keras treats all equally
            class_weight = filtered_class_weight if filtered_class_weight else None

        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=deepcopy(callbacks) if callbacks else [],
            class_weight=class_weight,
        )
        return self

    def _check_callbacks(self, callbacks):
        from tensorflow import keras

        # if user hasn't provided a custom ReduceLROnPlateau via init already,
        # add the default from literature
        if callbacks is None:
            callbacks = []

        if not any(
            isinstance(callback, keras.callbacks.ReduceLROnPlateau)
            for callback in callbacks
        ):
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=0.0001
            )
            callbacks = callbacks + [reduce_lr]
        return callbacks

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        from sktime.utils.dependencies import _check_soft_dependencies

        param1 = {
            "n_epochs": 10,
            "batch_size": 4,
        }

        param2 = {
            "n_epochs": 12,
            "batch_size": 6,
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
