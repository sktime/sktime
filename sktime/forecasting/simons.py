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
        See the Simons model card and references [1]_ for available checkpoints.
    revision : str or None, default=None
        Hugging Face revision to load.
    device : str, default="cpu"
        Device on which to load the model, for example ``"cpu"`` or ``"cuda"``.

    References
    ----------
    .. [1] https://huggingface.co/collections/abel-lab/simons-1
    .. [2] https://github.com/Abel-ai-lab/Simons
    .. [3] https://abel.ai/research/introducing-simons/

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.simons import SimonsForecaster
    >>> forecaster = SimonsForecaster()  # doctest: +SKIP
    >>> forecaster.fit(load_airline())  # doctest: +SKIP
    """

    _tags = {
        "category": ["deep_learning", "foundation_model"],
        "authors": ["saptadeep12", "Shu-Wan", "ruizhu-lgtm"],
        "maintainers": ["saptadeep12"],
        "capability:multivariate": True,
        "capability:exogenous": True,
        "capability:missing_values": True,
        "capability:pred_int": True,
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:non_contiguous_X": False,
        "capability:categorical_in_X": True,
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
        "tests:skip_by_name": ["test_get_test_params_coverage"],
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
        self.context_ = y.copy()
        self.X_ = None if X is None else X.copy()
        self.model_ = self._load_model()
        return self

    def _update(self, y, X=None, update_params=True):
        """Extend the stored history without refitting the zero-shot model."""
        from sktime.datatypes import update_data

        self.context_ = update_data(self.context_, y)
        if self.X_ is None:
            if X is not None:
                raise ValueError(
                    "X was provided during update, but no X was provided during fit."
                )
        else:
            if X is None:
                raise ValueError(
                    "X was provided during fit and is required during update."
                )
            if not X.columns.equals(self.X_.columns):
                raise ValueError(
                    "X columns during update must match the columns used during fit."
                )
            self.X_ = update_data(self.X_, X)
        return self

    def _predict(self, fh, X=None):
        """Return point forecasts from Simons."""
        from sktime.libs.simons import QUANTILES

        predictions, index = self._run_forecast(fh, X=X)
        values = predictions[:, :, QUANTILES.index(0.5)].T
        return pd.DataFrame(values, index=index, columns=self.context_.columns)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Return quantile forecasts, interpolating Simons' native quantiles.

        Requested levels in the native range [0.1, 0.9] are linearly
        interpolated from Simons' nine output quantiles. Levels outside that
        range are clamped to the nearest supported endpoint with a warning.
        """
        from sktime.libs.simons import QUANTILES

        if alpha is None:
            alpha = [0.1, 0.5, 0.9]
        if np.isscalar(alpha):
            alpha = [alpha]
        alpha = [float(value) for value in alpha]
        supported = np.asarray(QUANTILES, dtype=np.float64)
        effective_alpha = np.clip(alpha, supported[0], supported[-1])
        if not np.allclose(alpha, effective_alpha):
            warnings.warn(
                "Simons supports quantiles only in [0.1, 0.9]; "
                "out-of-range levels were clamped to the nearest endpoint.",
                UserWarning,
                stacklevel=2,
            )

        predictions, index = self._run_forecast(fh, X=X)
        values = np.apply_along_axis(
            lambda row: np.interp(effective_alpha, supported, row),
            axis=-1,
            arr=predictions,
        ).transpose(1, 0, 2)
        values = values.reshape(len(index), -1)
        columns = pd.MultiIndex.from_product([self.context_.columns, alpha])
        return pd.DataFrame(values, index=index, columns=columns)

    def _predict_proba(self, fh, X=None, marginal=True):
        """Return a histogram distribution from Simons' native quantiles."""
        from skpro.distributions import HistogramQPD

        from sktime.libs.simons import QUANTILES

        predictions, index = self._run_forecast(fh, X=X)
        levels = np.asarray(QUANTILES, dtype=np.float64)
        row_index = pd.MultiIndex.from_product([levels, index])
        quantiles = pd.DataFrame(
            predictions.transpose(2, 1, 0).reshape(-1, predictions.shape[0]),
            index=row_index,
            columns=self.context_.columns,
        )
        return HistogramQPD(
            quantiles, tails="mass", index=index, columns=self.context_.columns
        )

    def _run_forecast(self, fh, X=None):
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

        from sktime.forecasting.base import ForecastingHorizon
        from sktime.libs.simons import prepare_covariates

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
        history_encoded = self.X_.copy()
        future_encoded = future.copy()
        for column in self.X_.columns:
            history_column = self.X_[column]
            future_column = future[column]
            is_categorical = any(
                isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object
                for series in (history_column, future_column)
            )
            if not is_categorical:
                continue

            categories = sorted(
                {
                    str(value)
                    for series in (history_column, future_column)
                    for value in series
                    if not pd.isna(value)
                }
            )
            category_codes = {value: code for code, value in enumerate(categories)}
            for encoded_frame, series in (
                (history_encoded, history_column),
                (future_encoded, future_column),
            ):
                encoded_frame[column] = pd.Series(
                    [
                        (
                            np.nan
                            if pd.isna(value)
                            else category_codes.get(str(value), np.nan)
                        )
                        for value in series
                    ],
                    index=series.index,
                    dtype=np.float64,
                )

        encoded = pd.concat([history_encoded, future_encoded])
        values = prepare_covariates(
            length=len(self.context_),
            horizon=horizon,
            future=encoded,
        )
        if values is None:
            return None
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
        return [{"model_path": "abel-lab/Simons-25M"}]


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
