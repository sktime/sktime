# -*- coding: utf-8 -*-
"""
Abstract base class for the Keras neural network classifiers.

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags, _predict and _predict_proba
"""
__author__ = ["James-Large", "ABostrom", "TonyBagnall"]
__all__ = ["BaseDeepClassifier"]

import numpy as np
from abc import abstractmethod
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sktime.classification.base import BaseClassifier


class BaseDeepClassifier(BaseClassifier):
    """Abstract base class for deep learning time series classifiers.

    The base classifier provides a deep learning default method for
    _predict and _predict_proba, and provides a new abstract method for building a
    model.

    Parameters
    ----------
    batch_size : int, default = 40
        training batch size for the model
    Arguments
    ---------
    self.model = None

    """
    _tags = {"X_inner_mtype": "numpy3D"}

    def __init__(self, batch_size=40):
        self.batch_size = batch_size
        self.model_ = None

    @abstractmethod
    def build_model(self, input_shape, n_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for
        training

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
        ...

    @abstractmethod
    def _fit(self, X, y):
        """Pass through abstract method from BaseClassifier."""
        ...

    def _predict(self, X, **kwargs):
        probs = _predict_proba(self, X,**kwargs)
        return probs # Max

    def _predict_proba(self, **kwargs):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        input_checks: boolean
            whether to check the X parameter
        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """

        probs = self.model_.predict(X, self.batch_size, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])

        return probs

    def convert_y(self, y, label_encoder=None, onehot_encoder=None):
        if (label_encoder is None) and (onehot_encoder is None):
            # make the encoders and store in self
            self.label_encoder = LabelEncoder()
            self.onehot_encoder = OneHotEncoder(sparse=False,
                                                categories="auto")
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