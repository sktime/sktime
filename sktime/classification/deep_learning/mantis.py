"""Mantis Time Series Classification Foundation Model."""

__all__ = ["MantisClassifier"]

import numpy as np

from sktime.classification.base import BaseClassifier
from sktime.utils.dependencies import _check_estimator_deps


class MantisClassifier(BaseClassifier):
    """Mantis Foundation Model for Time Series Classification.

    Mantis is a Vision Transformer (ViT) based foundation model for time series
    classification that can be used for feature extraction or fine-tuning.

    Parameters
    ----------
    pretrained : bool, default=True
        Whether to load pretrained weights.
    device : str, default="cpu"
        Device used for training ("cpu" or "cuda").
    n_epochs : int, default=50
        Number of training epochs.
    batch_size : int, default=32
        Batch size.
    lr : float, default=1e-4
        Learning rate.
    verbose : bool, default=False
        Print training progress.
    """

    _tags = {
        "authors": ["sktime developers"],
        "maintainers": ["sktime developers"],
        "python_dependencies": "mantis-tsfm",
        "capability:multivariate": True,
        "capability:predict_proba": True,
    }

    def __init__(
        self,
        pretrained=True,
        device="cpu",
        n_epochs=50,
        batch_size=32,
        lr=1e-4,
        verbose=False,
    ):
        self.pretrained = pretrained
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

        self.model_ = None

        super().__init__()

    def _fit(self, X, y):
        """Fit the Mantis model."""
        _check_estimator_deps(self)

        try:
            from mantis_tsfm import MantisModel
        except ImportError:
            raise ImportError(
                "mantis-tsfm is required. Install with: pip install mantis-tsfm"
            )

        if self.pretrained:
            try:
                self.model_ = MantisModel.from_pretrained("mantis-base")
            except Exception:
                if self.verbose:
                    print("Failed loading pretrained weights. Initializing new model.")
                self.model_ = MantisModel()
        else:
            self.model_ = MantisModel()

        try:
            self.model_.fit(
                X,
                y,
                epochs=self.n_epochs,
                batch_size=self.batch_size,
                learning_rate=self.lr,
                device=self.device,
                verbose=self.verbose,
            )
        except Exception as e:
            raise RuntimeError(f"Mantis training failed: {e}")

        return self

    def _predict(self, X):
        """Predict class labels."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        preds = self.model_.predict(X)
        preds = np.asarray(preds)

        if preds.ndim > 1:
            preds = np.argmax(preds, axis=1)

        if np.issubdtype(preds.dtype, np.integer):
            preds = self.classes_[preds]

        return preds

    def _predict_proba(self, X):
        """Predict class probabilities."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")

        probs = self.model_.predict_proba(X)
        probs = np.asarray(probs)

        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])

        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return minimal parameters for CI testing."""
        return {
            "n_epochs": 1,
            "batch_size": 4,
            "lr": 1e-3,
            "pretrained": False,
        }