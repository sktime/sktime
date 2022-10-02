# -*- coding: utf-8 -*-
"""
Abstract base class for the Keras neural network regressors.

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags and _predict
"""

__author__ = ["AurumnPegasus", "achieveordie"]
__all__ = ["BaseDeepRegressor"]

from abc import ABC, abstractmethod

import numpy as np

from sktime.regression.base import BaseRegressor


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
        "python_dependencies": "tensorflow",
    }

    def __init__(self, batch_size=40):
        super(BaseDeepRegressor, self).__init__()

        self.batch_size = batch_size
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
        y_pred = np.squeeze(y_pred, axis=-1)
        return y_pred

    def __getstate__(self):
        """Get Dict config that will be used when a serialization method is called.

        Returns
        -------
        copy : dict, the config to be serialized
        """
        copy = self.__dict__.copy()
        check_before_deletion = ["model_", "history", "optimizer_"]
        # if attribute "optimizer" is not None, then it must
        # have been supplied by the user and will be equal to "optimizer_"
        # delete it normally with other non-serializable attributes
        if copy["optimizer"] is not None:
            check_before_deletion.append("optimizer")
        # if it is None, then save it as 0, so it can be
        # later correctly restored as None
        else:
            copy["optimizer"] = 0
        for attribute in check_before_deletion:
            if copy.get(attribute) is not None:
                del copy[attribute]
        return copy

    def __setstate__(self, state):
        """Magic method called during deserialization.

        Parameters
        ----------
        state : dict, as returned from __getstate__(), used for correct deserialization

        Returns
        -------
        -
        """
        self.__dict__ = state
        self.__dict__["model_"] = self.model_
        # Having 0 as value implies "optimizer" attribute was None
        # as per __getstate__()
        if self.__dict__.get("optimizer") == 0:
            self.__dict__["optimizer"] = None
        else:
            self.__dict__["optimizer"] = self.model_.optimizer
        self.__dict__["optimizer_"] = self.model_.optimizer
        self.__dict__["history"] = self.historyy
