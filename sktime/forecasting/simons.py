# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Simons forecaster for ``sktime``."""

__all__ = ["SimonsForecaster"]

import warnings

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class SimonsForecaster(BaseForecaster):
    """Simons zero-shot probabilistic forecaster.

    Simons is a pretrained time-series foundation model for zero-shot
    forecasting. ``fit`` stores the observed history and loads the selected
    checkpoint; it does not train or update model weights.

    ``X`` is interpreted as known-future covariates. It must be supplied
    during ``fit`` and during ``predict`` and cover every step through the
    maximum requested forecasting horizon. Past-only covariates are not
    supported.

    The Simons inference code is licensed under Apache-2.0. Simons model
    weights are licensed separately under OpenMDW-1.1 and are downloaded at
    runtime; weights are not bundled with ``sktime``.

    Parameters
    ----------
    model_path : str, default="abel-lab/Simons-60M"
        Hugging Face model identifier or local checkpoint path.
    revision : str or None, default=None
        Hugging Face revision to load.
    device : str, default="cpu"
        Device on which to load the model, for example ``"cpu"`` or ``"cuda"``.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.simons import SimonsForecaster
    >>> forecaster = SimonsForecaster()  # doctest: +SKIP
    >>> forecaster.fit(load_airline())  # doctest: +SKIP
    """

    _tags = {
        "category": ["deep_learning", "foundation_model"],
        "authors": ["starrynights07"],
        "maintainers": ["starrynights07"],
        "capability:multivariate": True,
        "capability:exogenous": True,
        "capability:missing_values": True,
        "capability:pred_int": True,
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:non_contiguous_X": False,
        "capability:categorical_in_X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "python_dependencies": [
            "torch>=2.5",
            "numpy>=1.24",
            "safetensors>=0.4",
            "huggingface_hub>=0.23",
        ],
        "tests:libs": ["sktime.libs.simons"],
        "tests:specific": ["sktime.forecasting.tests.test_simons"],
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path: str = "abel-lab/Simons-60M",
        revision: str | None = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.revision = revision
        self.device = device

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Store the history and load the pretrained Simons model."""
        if X is not None and not X.index.equals(y.index):
            raise ValueError("X and y must have the same index during fit.")
        self.context_ = y.copy()
        self.X_ = None if X is None else X.copy()
        self.model_ = self._load_model()
        return self

    def _predict(self, fh, X=None):
        """Return point forecasts from Simons."""
        predictions, index = self._forecast_quantiles(fh, X=X)
        values = predictions[:, :, 4].T
        return pd.DataFrame(values, index=index, columns=self.context_.columns)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Return quantile forecasts, interpolating Simons' native quantiles.

        Requested levels in the native range [0.1, 0.9] are linearly
        interpolated from Simons' nine output quantiles. Levels outside that
        range are clamped to the nearest supported endpoint with a warning.
        """
        if alpha is None:
            alpha = [0.1, 0.5, 0.9]
        if np.isscalar(alpha):
            alpha = [alpha]
        alpha = [float(value) for value in alpha]
        supported = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            dtype=np.float64,
        )
        effective_alpha = np.clip(alpha, supported[0], supported[-1])
        if not np.allclose(alpha, effective_alpha):
            warnings.warn(
                "Simons supports quantiles only in [0.1, 0.9]; "
                "out-of-range levels were clamped to the nearest endpoint.",
                UserWarning,
                stacklevel=2,
            )

        predictions, index = self._forecast_quantiles(fh, X=X)
        positions = (effective_alpha - supported[0]) / (supported[1] - supported[0])
        lower = np.floor(positions).astype(int)
        upper = np.ceil(positions).astype(int)
        weights = positions - lower
        values = np.stack(
            [
                predictions[:, :, low] * (1 - weight) + predictions[:, :, high] * weight
                for low, high, weight in zip(lower, upper, weights)
            ],
            axis=2,
        ).transpose(1, 0, 2)
        values = values.reshape(len(index), -1)
        columns = pd.MultiIndex.from_product([self.context_.columns, alpha])
        return pd.DataFrame(values, index=index, columns=columns)

    def _forecast_quantiles(self, fh, X=None):
        """Return selected Simons quantiles and their absolute forecast index."""
        import torch

        relative_fh = fh.to_relative(self.cutoff)
        relative_steps = np.asarray(relative_fh.to_numpy(), dtype=np.int64)
        horizon = int(relative_steps.max())

        history = np.array(self.context_, dtype=np.float32, copy=True).T
        history_tensor = torch.as_tensor(history, dtype=torch.float32).unsqueeze(0)
        covariates = self._build_covariates(X, horizon)
        if covariates is None:
            predictions = self.model_.forecast(history_tensor, horizon=horizon)
        else:
            predictions = self.model_.forecast(
                history_tensor,
                horizon=horizon,
                covariates=covariates,
            )
        predictions = predictions.detach().float().cpu().numpy()
        expected_shape = (1, history.shape[0], horizon, 9)
        if predictions.shape != expected_shape:
            raise ValueError(
                "Simons returned quantiles with unexpected shape "
                f"{predictions.shape}; expected {expected_shape}."
            )

        selected = predictions[0][:, relative_steps - 1, :]
        index = fh.to_absolute(self.cutoff).to_pandas()
        return selected, index

    def _build_covariates(self, X, horizon):
        """Build a Simons covariate tensor from history and known future X."""
        import torch

        if self.X_ is None:
            if X is not None:
                raise ValueError(
                    "X was provided during predict, but no X was provided during fit. "
                    "Known-future covariates require history values during fit."
                )
            return None
        if X is None:
            raise ValueError(
                "X was provided during fit and is required during predict for "
                "known-future covariates."
            )
        if not X.columns.equals(self.X_.columns):
            raise ValueError(
                "X columns during predict must match the columns used during fit."
            )
        if len(self.X_.columns) > 16:
            raise ValueError("Simons supports at most 16 covariate columns.")

        from sktime.forecasting.base import ForecastingHorizon

        future_index = (
            ForecastingHorizon(range(1, horizon + 1))
            .to_absolute(self.cutoff)
            .to_pandas()
        )
        missing = future_index.difference(X.index)
        if len(missing) > 0:
            raise ValueError(
                "X during predict must contain every future row through the maximum "
                f"forecast horizon; missing {len(missing)} row(s)."
            )
        future = X.loc[future_index]
        try:
            history = np.array(self.X_, dtype=np.float32, copy=True).T
            future = np.array(future, dtype=np.float32, copy=True).T
        except (TypeError, ValueError) as exc:
            raise ValueError("Simons covariates must be numeric.") from exc
        values = np.concatenate([history, future], axis=1)
        return torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)

    def _load_model(self):
        """Load the checkpoint or retrieve it from the shared model cache."""
        self.model_ = _CachedSimons(
            key=self._get_unique_key(),
            model_path=self.model_path,
            revision=self.revision,
            device=self.device,
        ).load()
        return self.model_

    def _get_unique_key(self):
        """Return the cache key for this model configuration."""
        return (self.model_path, self.revision, self.device)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"model_path": "abel-lab/Simons-25M"},
            {"model_path": "abel-lab/Simons-25M", "device": "cpu"},
        ]


@_multiton
class _CachedSimons:
    """Multiton-backed cache wrapper for a loaded Simons model."""

    def __init__(self, key, model_path, revision, device):
        self.key = key
        self.model_path = model_path
        self.revision = revision
        self.device = device
        self.model_ = None

    def load(self):
        """Load the model on the requested device once."""
        if self.model_ is not None:
            return self.model_

        from sktime.libs.simons import Simons

        self.model_ = Simons.from_pretrained(
            self.model_path,
            device=self.device,
            revision=self.revision,
        )
        return self.model_
