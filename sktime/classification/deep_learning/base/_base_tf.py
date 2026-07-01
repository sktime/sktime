"""Abstract base class for the Keras neural network classifiers.

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags, _predict and _predict_proba
"""

__all__ = ["BaseDeepClassifier"]

from abc import abstractmethod

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_random_state

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

    Attributes
    ----------
    self.model_ - the fitted DL model
    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
        "python_dependencies": "tensorflow",
        "property:randomness": "stochastic",
        "capability:random_state": True,
        "serialization:native_artifacts": ("model_",),
        "serialization:skip": ("history", "optimizer", "optimizer_"),
        "authors": [
            "James-Large",
            "ABostrom",
            "TonyBagnall",
            "aurunmpegasus",
            "achieveordie",
            "noxthot",
        ],
    }

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
        """Summary function to return the losses/metrics for model fit.

        Returns
        -------
        history: dict or None,
            Dictionary containing model's train/validation losses and metrics
        """
        history = getattr(self, "history", None)
        return history.history if history is not None else None

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

    def _convert_y_to_keras(self, y):
        """Convert y to required Keras format."""
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        y = y.reshape(len(y), 1)

        # in sklearn 1.2, sparse was renamed to sparse_output
        if _check_soft_dependencies("scikit-learn>=1.2", severity="none"):
            sparse_kw = {"sparse_output": False}
        else:
            sparse_kw = {"sparse": False}

        self.onehot_encoder = OneHotEncoder(categories="auto", **sparse_kw)
        # categories='auto' to get rid of FutureWarning
        y = self.onehot_encoder.fit_transform(y)
        return y
