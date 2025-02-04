"""Abstract base class for the Keras neural network classifiers.

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags, _predict and _predict_proba
"""

__author__ = ["James-Large", "ABostrom", "TonyBagnall", "aurunmpegasus", "achieveordie"]
__all__ = ["BaseDeepClassifier"]

from abc import abstractmethod

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_random_state

from sktime.base._base_panel import DeepSerializationMixin
from sktime.classification.base import BaseClassifier
from sktime.utils.dependencies import _check_soft_dependencies


class BaseDeepClassifier(BaseClassifier, DeepSerializationMixin):
    """Abstract base class for deep learning time series classifiers.

    Parameters
    ----------
    batch_size : int, default=40
        Training batch size for the model.

    Attributes
    ----------
    self.model_ : the fitted DL model
    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
        "python_dependencies": "tensorflow",
    }

    @abstractmethod
    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained Keras model ready for training.

        Parameters
        ----------
        input_shape : tuple
            Shape of the data fed into the input layer.
        n_classes : int
            Number of classes (size of the output layer).

        Returns
        -------
        A compiled Keras Model.
        """
        ...

    def summary(self):
        """Return a summary of the model's training losses/metrics."""
        return self.history.history if self.history is not None else None

    def _predict_proba(self, X, **kwargs):
        """Predict class probability estimates for all cases in X."""
        X = X.transpose((0, 2, 1))
        probs = self.model_.predict(X, self.batch_size, **kwargs)
        if probs.shape[1] == 1:  # binary classification
            probs = np.hstack([1 - probs, probs])
        probs = probs / probs.sum(axis=1, keepdims=1)
        return probs

    def _predict(self, X, **kwargs):
        """Predict classes for all cases in X."""
        probs = self._predict_proba(X, **kwargs)
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in probs
            ]
        )

    def _convert_y_to_keras(self, y):
        """Convert y into the format required by Keras."""
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        y_encoded = y_encoded.reshape(len(y_encoded), 1)
        # Use the appropriate parameter depending on scikit-learn version
        if _check_soft_dependencies("scikit-learn>=1.2", severity="none"):
            sparse_kw = {"sparse_output": False}
        else:
            sparse_kw = {"sparse": False}
        self.onehot_encoder = OneHotEncoder(categories="auto", **sparse_kw)
        y_keras = self.onehot_encoder.fit_transform(y_encoded)
        return y_keras
