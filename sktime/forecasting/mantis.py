# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mantis embedding-based forecaster."""

__author__ = ["vedantag17"]
__all__ = ["MantisForecaster"]

from copy import deepcopy

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class MantisForecaster(BaseForecaster):
    """Forecaster using Mantis time-series foundation model embeddings.

    Mantis is primarily a time-series classification foundation model. This
    forecaster uses its frozen backbone as a feature extractor on rolling history
    windows, then fits a regression model to predict the next value. Multi-step
    forecasts are generated recursively.

    Parameters
    ----------
    checkpoint : str or None, default="paris-noah/MantisV2"
        Hugging Face checkpoint to load via Mantis ``from_pretrained``.
        If None, use a randomly initialized Mantis backbone.
    model_version : {"v1", "v2"}, default="v2"
        Mantis architecture version. Use "v1" for "paris-noah/Mantis-8M" and
        "paris-noah/MantisPlus"; use "v2" for "paris-noah/MantisV2".
    context_length : int, default=512
        Number of most recent observations used for each supervised window.
    seq_len : int, default=512
        Length passed to Mantis. If different from ``context_length``, windows
        are resized with linear interpolation.
    regressor : sklearn regressor or None, default=None
        Regression model trained on Mantis embeddings. If None, ``Ridge()`` is used.
    batch_size : int, default=256
        Batch size for Mantis embedding extraction.
    device : str, default="auto"
        Torch device. If "auto", use CUDA when available, otherwise CPU.

    References
    ----------
    .. [1] Feofanov et al., "Mantis: Lightweight Calibrated Foundation Model
       for User-Friendly Time Series Classification", 2025.
       https://arxiv.org/abs/2502.15637

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.mantis import MantisForecaster
    >>> y = load_airline()
    >>> forecaster = MantisForecaster(context_length=24)
    >>> forecaster.fit(y)  # doctest: +SKIP
    MantisForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["vedantag17"],
        "python_dependencies": ["mantis-tsfm>=1.0.0", "torch"],
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:exogenous": False,
        "capability:insample": False,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "property:randomness": "stochastic",
        "serialization:skip": ("trainer_",),
        # CI and testing tags
        # -------------------
        "tests:vm": True,
        # relevant issue: https://github.com/sktime/sktime/issues/10491
        # deepcopy fails during `update_predict(..., reset_forecaster=False)`
        "tests:skip_by_name": ["test_update_predict_predicted_index"],
    }

    def __init__(
        self,
        checkpoint="paris-noah/MantisV2",
        model_version="v2",
        context_length=512,
        seq_len=512,
        regressor=None,
        batch_size=256,
        device="auto",
        ignore_deps=False,
    ):
        self.checkpoint = checkpoint
        self.model_version = model_version
        self.context_length = context_length
        self.seq_len = seq_len
        self.regressor = regressor
        self.batch_size = batch_size
        self.device = device
        self.ignore_deps = ignore_deps

        self.trainer_ = None

        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        if self.ignore_deps:
            self.set_tags(python_dependencies=[])

    def __getstate__(self):
        """Return state for pickling, excluding unpickleable trainer."""
        state = self.__dict__.copy()
        if "trainer_" in state:
            state["trainer_"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled state dictionary."""
        self.__dict__.update(state)

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        y_arr = self._coerce_y_to_numpy(y)
        self._is_series_ = isinstance(y, pd.Series)
        self._y_name_ = y.name if self._is_series_ else None
        self._y_columns_ = None if self._is_series_ else y.columns
        self._n_channels_ = y_arr.shape[1]

        if self.context_length < 1:
            raise ValueError("context_length must be at least 1.")
        if self.seq_len < 1:
            raise ValueError("seq_len must be at least 1.")
        if self.seq_len % 32 != 0:
            raise ValueError("seq_len must be a multiple of 32 for Mantis.")
        if len(y_arr) <= self.context_length:
            raise ValueError(
                "MantisForecaster needs more observations than context_length. "
                f"Found {len(y_arr)} observations and context_length="
                f"{self.context_length}."
            )

        self._device_ = self._check_device()
        self.trainer_ = self._load_trainer()

        X_train, y_train = self._make_tabular_training_data(y_arr)
        embeddings = self._transform_windows(X_train)

        regressor = Ridge() if self.regressor is None else deepcopy(self.regressor)
        if self._n_channels_ > 1:
            regressor = MultiOutputRegressor(regressor)
        self.regressor_ = regressor.fit(embeddings, y_train)

        self._last_window_ = y_arr[-self.context_length :].copy()
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon."""
        fh_relative = fh.to_relative(self.cutoff).to_numpy()
        if np.any(fh_relative <= 0):
            raise NotImplementedError(
                "MantisForecaster only supports strictly out-of-sample forecasting."
            )

        max_horizon = int(np.max(fh_relative))
        recursive_preds = self._predict_recursive(max_horizon)
        y_pred = recursive_preds[fh_relative.astype(int) - 1]

        index = fh.to_absolute(self.cutoff).to_pandas()
        if self._is_series_:
            return pd.Series(y_pred.ravel(), index=index, name=self._y_name_)
        return pd.DataFrame(y_pred, index=index, columns=self._y_columns_)

    def _make_tabular_training_data(self, y):
        """Create rolling windows and one-step targets."""
        windows = []
        targets = []
        for cutoff in range(self.context_length, len(y)):
            windows.append(y[cutoff - self.context_length : cutoff].T)
            targets.append(y[cutoff])
        return (
            np.asarray(windows, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
        )

    def _predict_recursive(self, max_horizon):
        """Predict recursively up to ``max_horizon``."""
        window = self._last_window_.copy()
        preds = []

        for _ in range(max_horizon):
            X_window = window.T[None, :, :].astype(np.float32)
            embedding = self._transform_windows(X_window)
            next_value = np.asarray(
                self.regressor_.predict(embedding), dtype=np.float32
            )
            next_value = next_value.reshape(1, self._n_channels_)
            preds.append(next_value[0])
            window = np.vstack([window[1:], next_value])

        return np.asarray(preds)

    def _transform_windows(self, X):
        """Transform windows to Mantis embeddings."""
        self._ensure_trainer_loaded()
        X = self._resize_windows(X)
        return self.trainer_.transform(X, batch_size=self.batch_size)

    def _resize_windows(self, X):
        """Resize windows to the Mantis sequence length if needed."""
        if X.shape[-1] == self.seq_len:
            return X

        import torch
        import torch.nn.functional as F

        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_resized = F.interpolate(
            X_tensor, size=self.seq_len, mode="linear", align_corners=False
        )
        return X_resized.numpy()

    def _get_trainer_kwargs(self):
        return {
            "checkpoint": self.checkpoint,
            "model_version": self.model_version,
            "seq_len": self.seq_len,
            "device": self._device_,
        }

    def _get_unique_key(self):
        kwargs = self._get_trainer_kwargs()
        return str(sorted(kwargs.items()))

    def _load_trainer(self):
        """Load Mantis network and wrap it in MantisTrainer."""
        if not self.ignore_deps:
            _check_soft_dependencies("mantis-tsfm>=1.0.0", "torch", severity="error")
        return _CachedMantis(
            key=self._get_unique_key(),
            mantis_kwargs=self._get_trainer_kwargs(),
        ).load_trainer()

    def _ensure_trainer_loaded(self):
        """Reload trainer if needed after unpickling."""
        if not hasattr(self, "trainer_") or self.trainer_ is None:
            if hasattr(self, "_is_fitted") and self._is_fitted:
                self.trainer_ = self._load_trainer()

    def _check_device(self):
        """Resolve the torch device."""
        if not self.ignore_deps:
            _check_soft_dependencies("torch", severity="error")
        try:
            import torch

            if self.device == "auto":
                return "cuda" if torch.cuda.is_available() else "cpu"
            return self.device
        except ImportError:
            return "cpu"

    @staticmethod
    def _coerce_y_to_numpy(y):
        """Convert sktime y input to 2D numpy array."""
        if isinstance(y, pd.Series):
            return y.to_numpy(dtype=np.float32).reshape(-1, 1)
        return y.to_numpy(dtype=np.float32)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {
                "checkpoint": None,
                "context_length": 4,
                "seq_len": 128,
                "batch_size": 4,
                "device": "cpu",
            },
            {
                "checkpoint": None,
                "model_version": "v1",
                "context_length": 6,
                "seq_len": 128,
                "regressor": Ridge(alpha=0.5),
                "batch_size": 4,
                "device": "cpu",
            },
        ]


@_multiton
class _CachedMantis:
    """Cached Mantis model to ensure only one instance exists in memory.

    Mantis is a zero-shot model and immutable, so sharing the same instance
    across multiple uses has no side effects.
    """

    def __init__(self, key, mantis_kwargs):
        self.key = key
        self.mantis_kwargs = mantis_kwargs
        self.trainer = None

    def load_trainer(self):
        """Load Mantis network and wrap it in MantisTrainer."""
        if self.trainer is not None:
            return self.trainer

        from mantis.architecture import MantisV1, MantisV2
        from mantis.trainer import MantisTrainer

        model_version = self.mantis_kwargs["model_version"]
        seq_len = self.mantis_kwargs["seq_len"]
        device = self.mantis_kwargs["device"]
        checkpoint = self.mantis_kwargs["checkpoint"]

        if model_version == "v1":
            network = MantisV1(seq_len=seq_len, device=device)
        elif model_version == "v2":
            network = MantisV2(device=device)
        else:
            raise ValueError("model_version must be either 'v1' or 'v2'.")

        if checkpoint is not None:
            network = network.from_pretrained(checkpoint)

        self.trainer = MantisTrainer(device=device, network=network)
        return self.trainer
