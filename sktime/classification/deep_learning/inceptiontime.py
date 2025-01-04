from tensorflow.keras import layers, models, optimizers, callbacks
from sktime.utils.validation._dependencies import _check_dl_dependencies
from sktime.networks.inceptiontime import InceptionTimeNetwork
from sktime.classification.deep_learning.base import BaseDeepClassifier
from sklearn.utils import check_random_state
from copy import deepcopy

class InceptionTimeClassifier(BaseDeepClassifier):
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
    ):
        _check_dl_dependencies(severity="error")

        self.batch_size = batch_size
        self.bottleneck_size = bottleneck_size
        self.callbacks = callbacks
        self.depth = depth
        self.kernel_size = kernel_size
        self.loss = loss
        self.metrics = metrics if metrics is not None else ["accuracy"]
        self.n_epochs = n_epochs
        self.n_filters = n_filters
        self.random_state = random_state
        self.use_bottleneck = use_bottleneck
        self.use_residual = use_residual
        self.verbose = verbose

        super().__init__()

        network_params = {
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
        """Construct a compiled, un-trained, keras model that is ready for training."""
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)
        output_layer = layers.Dense(n_classes, activation="softmax")(output_layer)

        model = models.Model(inputs=input_layer, outputs=output_layer)

        optimizer = optimizers.Adam()  # Updated to the new API
        model.compile(
            loss=self.loss,
            optimizer=optimizer,
            metrics=self.metrics,
        )

        return model

    def _check_callbacks(self, callbacks_list):
        """Ensure callbacks are compatible with Keras v3."""
        if callbacks_list is None:
            callbacks_list = []

        # Ensure ReduceLROnPlateau is added if not already present
        if not any(isinstance(cb, callbacks.ReduceLROnPlateau) for cb in callbacks_list):
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=1e-4
            )
            callbacks_list.append(reduce_lr)

        return callbacks_list

    def _fit(self, X, y):
        """Fit the classifier on the training set."""
        y_onehot = self._convert_y_to_keras(y)
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.model_ = self.build_model(self.input_shape, self.n_classes_)
        if self.verbose:
            self.model_.summary()

        callbacks_to_use = self._check_callbacks(self.callbacks)

        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=deepcopy(callbacks_to_use),
        )
        return self

    def _convert_y_to_keras(self, y):
        """Convert target labels to one-hot encoding for Keras."""
        from tensorflow.keras.utils import to_categorical

        return to_categorical(y, num_classes=self.n_classes_)

    def _predict(self, X):
        """Predict class probabilities for X."""
        X = X.transpose(0, 2, 1)
        return self.model_.predict(X)

    def get_test_params(self, parameter_set="default"):
        """Return testing parameters."""
        return {
            "n_epochs": 20,
            "batch_size": 4,
            "depth": 3,
        }
"""
InceptionTimeClassifier: A Deep Learning Classifier for Time Series Data

This code defines the `InceptionTimeClassifier`, a deep learning-based classifier for time-series 
data based on the InceptionTime architecture. Below are the steps to use and enhance this classifier:

1. **Test the Classifier**:
    - Load sample data using `sktime.datasets` or other sources.
    - Train the classifier using `.fit()` and validate predictions using `_predict()`.

    Example:
    ```python
    from sktime.datasets import load_basic_motions
    from sklearn.model_selection import train_test_split

    X, y = load_basic_motions(split="train", return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = InceptionTimeClassifier(n_epochs=50, batch_size=16, verbose=True)
    clf.fit(X_train, y_train)

    predictions = clf._predict(X_test)
    print(predictions)
    ```

2. **Debug Errors**:
    - Ensure compatibility between `sktime`, TensorFlow, and Keras versions.
    - Verify `_check_dl_dependencies` is correctly implemented.

3. **Add Unit Tests**:
    - Write test cases to validate the classifier's behavior (e.g., output shapes, model compilation).
    Example:
    ```python
    def test_inception_time_classifier():
        from sktime.datasets import load_basic_motions
        X, y = load_basic_motions(split="train", return_X_y=True)

        clf = InceptionTimeClassifier(n_epochs=2, batch_size=8, verbose=False)
        clf.fit(X, y)

        preds = clf._predict(X)
        assert preds.shape[0] == X.shape[0]
    ```
