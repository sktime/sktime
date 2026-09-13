# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tafsut foundation model forecaster."""

__all__ = ["TafsutForecaster"]
__author__ = ["Tafsut-FM"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class TafsutForecaster(BaseForecaster):
    """Zero-shot probabilistic forecaster using the Tafsut foundation model.

    Tafsut is a univariate, probabilistic time-series foundation model. It
    returns forecasts at nine quantile levels from 0.1 through 0.9 and does
    not support exogenous variables or in-sample predictions.

    Parameters
    ----------
    model_path : str or None, default="Tafsut-FM/tafsut-univariate-base"
        Local model directory or Hugging Face model repository identifier. If
        ``None``, initialize a random model from ``config``.
    config : dict, optional
        Configuration values for a model initialized with ``model_path=None``.
    device : str or torch.device, optional
        Device used for inference. If ``None``, Tafsut selects CUDA when
        available and CPU otherwise.
    revision : str, optional
        Hugging Face model revision.
    token : str or bool, optional
        Hugging Face authentication token.
    cache_dir : str, optional
        Directory used by Hugging Face Hub for model caching.
    local_files_only : bool, default=False
        If ``True``, do not download model files from Hugging Face Hub.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.forecasting.tafsut import TafsutForecaster
    >>> config = {
    ...     "context_length": 8,
    ...     "prediction_length": 4,
    ...     "input_patch_size": 2,
    ...     "output_patch_size": 2,
    ...     "input_patch_stride": 2,
    ...     "d_model": 8,
    ...     "d_kv": 2,
    ...     "d_ff": 16,
    ...     "num_layers": 1,
    ...     "num_heads": 4,
    ...     "dropout_rate": 0.0,
    ... }
    >>> forecaster = TafsutForecaster(
    ...     model_path=None, config=config, device="cpu"
    ... )
    >>> y = pd.Series(range(8))
    >>> _ = forecaster.fit(y, fh=[1, 2, 3, 4])
    >>> y_pred = forecaster.predict()
    """

    _tags = {
        "authors": ["tareq-si-salem", "aryamanDutta"],
        "maintainers": ["sktime developers"],
        "python_version": ">=3.10",
        "python_dependencies": [
            "tafsut>=0.1.0",
        ],
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:missing_values": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:unequal_length": True,
        "requires-fh-in-fit": False,
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path: str | None = "Tafsut-FM/tafsut-univariate-base",
        config: dict | None = None,
        device=None,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.revision = revision
        self.token = token
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._model = None
        super().__init__()

    def __getstate__(self):
        """Exclude the loaded model from serialized estimator state."""
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        """Restore serialized estimator state."""
        self.__dict__.update(state)

    def _get_model_key(self):
        return str(
            (
                self.model_path,
                self.config,
                self.device,
                self.revision,
                self.token,
                self.cache_dir,
                self.local_files_only,
            )
        )

    def _load_model(self):
        return _CachedTafsutModel(
            key=self._get_model_key(),
            model_path=self.model_path,
            config=self.config,
            device=self.device,
            revision=self.revision,
            token=self.token,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        ).load()

    def _fit(self, y, X=None, fh=None):
        self._model = self._load_model()
        self._context = y.copy()
        return self

    def _forecast(self, horizon):
        if self._model is None:
            self._model = self._load_model()
        from tafsut import forecast

        return forecast(
            self._model,
            self._context.to_numpy(dtype=np.float32),
            horizon=horizon,
        )

    def _get_relative_horizon(self, fh):
        relative = fh.to_relative(self.cutoff).to_numpy(dtype=int)
        if len(relative) == 0 or np.any(relative <= 0):
            raise NotImplementedError(
                "TafsutForecaster does not support in-sample predictions."
            )
        return relative

    def _get_quantiles(self):
        quantiles = np.asarray(self._model.cfg.quantiles, dtype=float)
        if quantiles.ndim != 1 or len(quantiles) == 0:
            raise ValueError("Tafsut model must define a non-empty quantile sequence.")
        if np.any(np.diff(quantiles) <= 0):
            raise ValueError("Tafsut model quantiles must be strictly increasing.")
        return quantiles

    @staticmethod
    def _to_numpy(values):
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values)

    def _get_forecast_values(self, fh):
        relative = self._get_relative_horizon(fh)
        output = self._to_numpy(self._forecast(int(np.max(relative))))
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError(
                "Tafsut forecast output must have shape (1, horizon, quantiles)."
            )
        if output.shape[2] != len(self._get_quantiles()):
            raise ValueError("Tafsut output quantile dimension does not match config.")
        return output[0, relative - 1], relative

    def _predict(self, fh, X=None):
        values, _ = self._get_forecast_values(fh)
        quantiles = self._get_quantiles()
        median_indices = np.flatnonzero(np.isclose(quantiles, 0.5))
        if len(median_indices) != 1:
            raise ValueError("Tafsut model must define exactly one 0.5 quantile.")
        median_index = int(median_indices[0])
        return pd.Series(
            values[:, median_index],
            index=fh.to_absolute(self.cutoff)._values,
            name=self._context.name,
        )

    def _predict_quantiles(self, fh, X=None, alpha=None):
        values, _ = self._get_forecast_values(fh)
        native_quantiles = self._get_quantiles()
        requested = (
            native_quantiles if alpha is None else np.asarray(alpha, dtype=float)
        )
        quantile_values = np.column_stack(
            [
                np.array(
                    [
                        np.interp(requested_quantile, native_quantiles, row)
                        for row in values
                    ]
                )
                for requested_quantile in requested
            ]
        )
        name = self._context.name if self._context.name is not None else 0
        columns = pd.MultiIndex.from_product([[name], requested])
        return pd.DataFrame(
            quantile_values,
            index=fh.to_absolute(self.cutoff)._values,
            columns=columns,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return parameter settings for estimator testing."""
        return [
            {
                "model_path": None,
                "config": {
                    "context_length": 8,
                    "prediction_length": 4,
                    "input_patch_size": 2,
                    "output_patch_size": 2,
                    "input_patch_stride": 2,
                    "d_model": 8,
                    "d_kv": 2,
                    "d_ff": 16,
                    "num_layers": 1,
                    "num_heads": 4,
                    "dropout_rate": 0.0,
                },
                "device": "cpu",
            }
        ]


@_multiton
class _CachedTafsutModel:
    """Cache loaded Tafsut models by their loading configuration."""

    def __init__(
        self,
        key,
        model_path,
        config,
        device,
        revision,
        token,
        cache_dir,
        local_files_only,
    ):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.revision = revision
        self.token = token
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.model = None

    def load(self):
        if self.model is None:
            from tafsut import TafsutConfig, TafsutModel

            if self.model_path is None:
                import torch

                config = TafsutConfig(**(self.config or {}))
                device = self.device
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                device = torch.device(device)
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA was requested but is not available")
                self.model = TafsutModel(config).to(device)
                self.model.eval()
            else:
                self.model = TafsutModel.from_pretrained(
                    self.model_path,
                    device=self.device,
                    revision=self.revision,
                    token=self.token,
                    cache_dir=self.cache_dir,
                    local_files_only=self.local_files_only,
                )
        return self.model
