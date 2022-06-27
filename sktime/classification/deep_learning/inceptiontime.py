# -*- coding: utf-8 -*-
"""InceptionTime for classification."""
__author__ = "James Large"
__all__ = ["InceptionTimeClassifier"]

from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.inceptiontime import InceptionTimeNetwork
from sktime.utils.data import check_and_clean_data, check_and_clean_validation_data
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies("tensorflow", severity="warning")


class InceptionTimeClassifier(BaseDeepClassifier):
    """InceptionTime Deep Learning Classifier.

    Parameters
    ----------
    nb_filters: int,
    use_residual: boolean,
    use_bottleneck: boolean,
    depth: int
    kernel_size: int, specifying the length of the 1D convolution
     window
    batch_size: int, the number of samples per gradient update.
    bottleneck_size: int,
    n_epochs: int, the number of epochs to train the model
    callbacks: list of tf.keras.callbacks.Callback objects
    random_state: int, seed to any needed random actions
    verbose: boolean, whether to output extra information
    model_name: string, the name of this model for printing and
     file writing purposes
    model_save_directory: string, if not None; location to save
     the trained keras model in hdf5 format

    Notes
    -----
    ..[1] Fawaz et. al, InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
    """

    def __init__(
        self,
        n_epochs=1500,
        batch_size=64,
        kernel_size=41 - 1,
        nb_filters=32,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        callbacks=None,
        random_state=0,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
    ):
        _check_dl_dependencies(severity="error")
        super(InceptionTimeClassifier, self).__init__()

        self.verbose = verbose

        # predefined
        self.batch_size = batch_size
        self.bottleneck_size = bottleneck_size
        self.callbacks = callbacks
        self.depth = depth
        self.kernel_size = kernel_size
        self.loss = loss
        self.metrics = metrics
        self.n_epochs = n_epochs
        self.nb_filters = nb_filters
        self.random_state = random_state
        self.use_bottleneck = use_bottleneck
        self.use_residual = use_residual
        self.verbose = verbose
        self._is_fitted = False
        self._network = InceptionTimeNetwork()

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

        output_layer = keras.layers.Dense(n_classes, activation="softmax")(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        # if user hasn't provided own metrics use accuracy
        if self.metrics is None:
            self.metrics = ["accuracy"]

        model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.Adam(),
            metrics=self.metrics,
        )

        # if user hasn't provided a custom ReduceLROnPlateau via init already,
        # add the default from literature
        if self.callbacks is None:
            self.callbacks = []

        if not any(
            isinstance(callback, keras.callbacks.ReduceLROnPlateau)
            for callback in self.callbacks
        ):
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=0.0001
            )
            self.callbacks.append(reduce_lr)

        return model

    def _fit(
        self, X, y, input_checks=True, validation_X=None, validation_y=None, **kwargs
    ):
        """Fit the classifier on the training set (X, y).

        Parameters
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
        self._callbacks = self.callbacks
        self.random_state = check_random_state(self.random_state)

        X = check_and_clean_data(X, y, input_checks=input_checks)
        y_onehot = self.convert_y_to_keras(y)

        validation_data = check_and_clean_validation_data(
            validation_X, validation_y, self.label_encoder, self.onehot_encoder
        )

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]

        if self.batch_size is None:
            self.batch_size = int(min(X.shape[0] / 10, 16))
        else:
            self.batch_size = self.batch_size

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
            validation_data=validation_data,
        )

        self._is_fitted = True

        return self
