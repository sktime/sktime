# -*- coding: utf-8 -*-
"""
Abstract base class for the Keras neural network classifiers.

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags, _predict and _predict_proba
"""
__author__ = ["James-Large", "ABostrom", "TonyBagnall", "achieveordie"]
__all__ = ["BaseDeepClassifier"]

from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.exceptions import NotFittedError


class BaseDeepClassifier(BaseClassifier, ABC):
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

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(self, batch_size=40, random_state=None):
        super(BaseDeepClassifier, self).__init__()

        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None

    @abstractmethod
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
        A compiled Keras Model
        """
        ...

    def summary(self):
        """
        Summary function to return the losses/metrics for model fit.

        Returns
        -------
        history: dict,
            Dictionary containing model's train/validation losses and metrics

        """
        return self.history.history

    def _predict(self, X, **kwargs):
        probs = self._predict_proba(X, **kwargs)
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in probs
            ]
        )

    def _predict_proba(self, X, **kwargs):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : an np.ndarray of shape = (n_instances, n_dimensions, series_length)
            The training input samples.         input_checks: boolean
            whether to check the X parameter

        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        # Transpose to work correctly with keras
        X = X.transpose((0, 2, 1))
        probs = self.model_.predict(X, self.batch_size, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        probs = probs / probs.sum(axis=1, keepdims=1)
        return probs

    def convert_y_to_keras(self, y):
        """Convert y to required Keras format."""
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        y = y.reshape(len(y), 1)
        self.onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
        # categories='auto' to get rid of FutureWarning
        y = self.onehot_encoder.fit_transform(y)
        return y

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
        self.__dict__["history"] = self.history

    def save(self, path=None):
        """Save serialized self to bytes-like object or to folder.

        Behaviour:
        if `path` is None, returns an in-memory serialized self
        if `path` is a file location, stores self at that location
        saved folder contains the following contents:
        metadata - contains class of self, i.e., type(self)
        object - serialized self. This class uses the default serialization (pickle)

        Parameters
        ----------
        path : None or folder location (str or Path)
            if None, self is saved to an in-memory object
            if folder location, self is saved to that folder location

        Returns
        -------
        if `path` is None - in-memory serialized self
        if `path` is file location - None
        """
        import pickle
        from pathlib import Path

        if path is None:
            return (type(self), pickle.dumps(self))

        if self.model_ is None:
            raise NotFittedError("Model not built yet, call it via `.build_model()`")

        path = Path(path)
        path.mkdir(exist_ok=True)

        self.model_.save(path / "keras/")
        with open(path / "history", "wb") as history_writer:
            pickle.dump(self.history.history, history_writer)

        pickle.dump(type(self), open(path / "_metadata", "wb"))
        pickle.dump(self, open(path / "_obj", "wb"))

    @classmethod
    def load_from_path(cls, serial):
        """Load object from file location.

        Parameters
        ----------
        serial : Name of the folder

        Returns
        -------
        deserialized self resulting in output at `path`, of `cls.save(path)`
        """
        import pickle

        from tensorflow import keras

        cls.model_ = keras.models.load_model(serial / "keras/")
        cls.history = keras.callbacks.History()
        cls.history.set_params(pickle.load(open(serial / "history", "rb")))

        return pickle.load(open(serial / "_obj", "rb"))
