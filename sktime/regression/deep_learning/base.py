"""Abstract base class for the Keras neural network regressors.

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags and _predict
"""

__author__ = ["AurumnPegasus", "achieveordie"]
__all__ = ["BaseDeepRegressor"]

from abc import abstractmethod

import numpy as np

from sktime.base._base_panel import DeepSerializationMixin
from sktime.regression.base import BaseRegressor


class BaseDeepRegressor(BaseRegressor, DeepSerializationMixin):
    """Abstract base class for deep learning time series regression.

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
    def build_model(self, input_shape, **kwargs):
        """Construct a compiled, un-trained Keras model ready for training.

        Parameters
        ----------
        input_shape : tuple
            Shape of the data fed into the input layer.

        Returns
        -------
        A compiled Keras Model.
        """
        ...

    def _predict(self, X, **kwargs):
        """Predict regression estimates for all cases in X."""
        X = X.transpose((0, 2, 1))
        y_pred = self.model_.predict(X, self.batch_size, **kwargs)
        y_pred = np.squeeze(y_pred, axis=-1)
        return y_pred
