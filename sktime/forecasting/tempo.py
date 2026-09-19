"""Implements the TEMPO zero-shot forecasting adapter."""

__author__ = ["idevede", "yongchand", "aryamanDutta"]
__all__ = ["TEMPOForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


@_multiton
class _CachedTEMPO:
    """Cache a single loaded TEMPO model instance per configuration."""

    def __init__(self, key, model_path, device, filename, cache_dir):
        self.key = key
        self.model_path = model_path
        self.device = device
        self.filename = filename
        self.cache_dir = cache_dir
        self.model = None

    def load_model(self):
        """Load the upstream TEMPO model once per configuration."""
        if self.model is not None:
            return self.model

        from tempo.models.TEMPO import TEMPO

        self.model = TEMPO.load_pretrained_model(
            device=self.device,
            repo_id=self.model_path,
            filename=self.filename,
            cache_dir=self.cache_dir,
        )
        return self.model


class TEMPOForecaster(BaseForecaster):
    """Zero-shot forecasting adapter for the TEMPO foundation model.

    This is a thin sktime wrapper around the upstream TEMPO implementation from the
    ``timeagi`` package. The upstream API exposes the model via
    ``from tempo.models.TEMPO import TEMPO`` and loads checkpoints using
    ``TEMPO.load_pretrained_model(...)``.

    Parameters
    ----------
    model_path : str, optional (default="Melady/TEMPO")
        Hugging Face repository identifier for the pretrained TEMPO model.
    filename : str, optional (default="TEMPO-80M_v1.pth")
        Model checkpoint filename.
    cache_dir : str, optional (default="./checkpoints/TEMPO_checkpoints")
        Local cache directory for downloaded model files.
    device : str or None, optional (default=None)
        Device to run inference on, e.g., "cpu" or "cuda".
    """

    _tags = {
        "authors": ["idevede", "yongchand", "aryamanDutta"],
        "maintainers": ["sktime developers"],
        "python_version": ">=3.10",
        "python_dependencies": ["timeagi"],
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "None",
        "capability:multivariate": False,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path="Melady/TEMPO",
        filename="TEMPO-80M_v1.pth",
        cache_dir="./checkpoints/TEMPO_checkpoints",
        device=None,
    ):
        self.model_path = model_path
        self.repo_id = model_path
        self.filename = filename
        self.cache_dir = cache_dir
        self.device = device

        super().__init__()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the forecaster."""
        return {"model_path": "Melady/TEMPO", "device": "cpu"}

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster by loading the upstream model instance."""
        if self.device is None:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = self.device

        self._y = y
        self._device = device
        self.model_ = self._load_model()
        return self

    def __getstate__(self):
        """Return state for pickling, excluding the loaded TEMPO model."""
        state = self.__dict__.copy()
        if "model_" in state:
            state["model_"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled dict."""
        self.__dict__.update(state)

    def _load_model(self):
        """Load TEMPO through the shared multiton cache."""
        return _CachedTEMPO(
            key=self._get_unique_tempo_key(),
            model_path=self.model_path,
            device=self._device,
            filename=self.filename,
            cache_dir=self.cache_dir,
        ).load_model()

    def _ensure_model_loaded(self):
        """Reload the cached TEMPO model after deserialization if necessary."""
        if not hasattr(self, "model_") or self.model_ is None:
            if self._is_fitted:
                self.model_ = self._load_model()

    def _get_unique_tempo_key(self):
        """Return a unique identifier for the cached TEMPO model."""
        return str(
            {
                "model_path": self.model_path,
                "filename": self.filename,
                "cache_dir": self.cache_dir,
                "device": self._device if hasattr(self, "_device") else self.device,
            }
        )

    def _predict(self, fh, X=None):
        """Predict for the absolute horizon requested by the user."""
        self._ensure_model_loaded()
        abs_idx = fh.to_absolute_index(self.cutoff)
        rel_idx = fh.to_relative(self.cutoff).to_numpy()
        pred_length = int(np.max(rel_idx))

        y_values = self._y.to_numpy()
        if np.asarray(y_values).ndim == 2 and y_values.shape[1] == 1:
            y_values = y_values.ravel()

        model_pred = self.model_.predict(y_values, pred_length=pred_length)
        pred = np.asarray(model_pred, dtype=float).reshape(-1)
        pred = pred[np.asarray(rel_idx, dtype=int) - 1]

        return pd.Series(pred, index=abs_idx, name=self._y.name)
