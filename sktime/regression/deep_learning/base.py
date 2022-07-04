# -*- coding: utf-8 -*-
"""
Abstract base class for the Keras neural network regressors.

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags and _predict
"""

__author__ = ["James-Large", "AurumnPegasus"]
__all__ = ["BaseDeepRegressor"]

from abc import ABC, abstractmethod

import numpy as np

# import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sktime.regression.base import BaseRegressor

# from sklearn.utils import check_random_state


class BaseDeepRegressor(BaseRegressor, ABC):
    """Abstract base class for deep learning time series regression.

    The base classifier provides a deep learning default method for
    _predict, and provides a new abstract method for building a
    model.

    Parameters
    ----------
    batch_size : int, default = 40
        training batch size for the model

    Arguments
    ---------
    self.model = None

    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
    }

    def __init__(self, batch_size=40, random_state=None):
        super(BaseDeepRegressor, self).__init__()

        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None

    @abstractmethod
    def build_model(self, input_shape, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        A compiled Keras Model
        """
        ...

    def _predict(self, X, **kwargs):
        """
        Find regression estimate for all cases in X.

        Parameters
        ----------
        X : an np.ndarray of shape = (n_instances, n_dimensions, series_length)
            The training input samples.

        Returns
        -------
        predictions : 1d numpy array
            array of predictions of each instance
        """
        X = X.transpose((0, 2, 1))
        y_pred = self.model_.predict(X, self.batch_size, **kwargs)
        # rng = check_random_state(self.random_state)
        y_pred = np.squeeze(y_pred, axis=-1)
        # if y_pred.ndim == 1:
        #     y_pred.ravel()
        return y_pred

    def convert_y_to_keras(self, y, label_encoder=None, onehot_encoder=None):
        """Convert y to required Keras format."""
        if (label_encoder is None) and (onehot_encoder is None):
            # make the encoders and store in self
            self.label_encoder = LabelEncoder()
            self.onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
            # categories='auto' to get rid of FutureWarning

            y = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
            self.n_classes_ = len(self.classes_)

            y = y.reshape(len(y), 1)
            y = self.onehot_encoder.fit_transform(y)
        else:
            y = label_encoder.fit_transform(y)
            y = y.reshape(len(y), 1)
            y = onehot_encoder.fit_transform(y)

        return y
