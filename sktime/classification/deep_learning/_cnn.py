# -*- coding: utf-8 -*-
"""Time Convolutional Neural Network (CNN) for classification"""

__author__ = ["JamesLarge", "TonyBagnall"]
__all__ = ["CNNClassifier"]

from sktime.classification.deep_learning import BaseDeepClassifier
from sktime.networks import CNNNetwork


from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data
from sklearn.utils import check_random_state
from tensorflow import keras


class CNNClassifier(BaseDeepClassifier, CNNNetwork):
    """Time Convolutional Neural Network (CNN).

    Parameters
    ----------
    nb_epochs: int, the number of epochs to train the model
    batch_size: int, the number of samples per gradient update.
    kernel_size: int, specifying the length of the 1D convolution
     window
    avg_pool_size: int, size of the average pooling windows
    n_conv_layers: int, the number of convolutional plus average
     pooling layers
    filter_sizes: int, array of shape = (nb_conv_layers)
    callbacks: list of tf.keras.callbacks.Callback objects
    random_state: int, or sklearn Random.state
    verbose: boolean, whether to output extra information
    model_name: string, the name of this model for printing and
    file writing purposes
    model_save_directory: string, if not None; location to save
    the trained keras model in hdf5 format

    Notes
    -----
    ..[1] Zhao et. al, Convolutional neural networks for
    time series classification, Journal of
    Systems Engineering and Electronics, 28(1):2017.

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py
    """

    def __init__(
            self,
            nb_epochs=2000,
            batch_size=16,
            kernel_size=7,
            avg_pool_size=3,
            n_conv_layers=2,
            filter_sizes=[6, 12],
            callbacks=None,
            random_state=0,
            verbose=False,
            model_name="cnn",
            model_save_directory=None,
    ):
        super(CNNClassifier, self).__init__(
            model_save_directory=model_save_directory,
            model_name=model_name)
        self.filter_sizes = filter_sizes
        self.nb_conv_layers = n_conv_layers
        self.avg_pool_size = avg_pool_size
        self.random_state = random_state
        self.kernel_size = kernel_size
        self.verbose = verbose
        self.callbacks = callbacks
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size

        self._is_fitted = False

    def build_model(self, input_shape, nb_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for
        training

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        nb_classes: int
            The number of classes, which shall become the size of the output
            layer

        Returns
        -------
        output : a compiled Keras Model
        """
        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=nb_classes, activation="sigmoid"
        )(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        return model

    def fit(self, X, y, input_checks=True, validation_X=None,
            validation_y=None, **kwargs):
        """
        Fit the classifier on the training set (X, y)

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
        self.random_state = check_random_state(self.random_state)

        if self.callbacks is None:
            self.callbacks = []

        X = check_and_clean_data(X, y, input_checks=input_checks)
        y_onehot = self.convert_y(y)

        validation_data = \
            check_and_clean_validation_data(validation_X, validation_y,
                                            self.label_encoder,
                                            self.onehot_encoder)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]

        self.model = self.build_model(self.input_shape, self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_data=validation_data,
        )

        self._is_fitted = True
        self.save_trained_model()

        return self
