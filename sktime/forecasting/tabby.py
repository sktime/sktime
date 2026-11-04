# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tabby forecaster for ``sktime``."""

__author__ = ["saptadeep12"]

__all__ = ["TabbyForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class TabbyForecaster(BaseForecaster):
    """Tabby probabilistic forecaster via Hugging Face ``transformers``.

    Tabby is a univariate time series foundation model. This forecaster will
    wrap its 99 quantile forecasts in the ``sktime`` forecasting interface;
    multivariate series are handled by ``sktime``'s vectorization.

    .. note::

        The Tabby weights are licensed under CC BY-NC 4.0 (non-commercial use)
        and are not bundled with ``sktime``. The model implementation is
        vendored under ``sktime.libs.tabby`` under its upstream Apache-2.0
        license.

    Parameters
    ----------
    model_path : str, default="paris-noah/Tabby"
        Hugging Face repository identifier or local path to a Tabby checkpoint.
    device : str, default="auto"
        Device on which to place the model, for example ``"cpu"``, ``"cuda"``,
        or ``"auto"``.
    dtype : torch.dtype or str, optional (default=None)
        Data type used for model loading, following the Transformers ``dtype``
        convention.
    revision : str or None, default=None
        Hugging Face repository revision to load.

    Notes
    -----
    - The forecast horizon is capped at 8,176 steps (the model's 8,192
      context minus its patch size of 16). Larger horizons raise an error.
    - The vendored model truncates history to its 8,192-position context and
      masks the requested future positions.
    - NaNs in the history are passed to the model as is; no imputation or
      scaling is applied.

    See Also
    --------
    BaseForecaster : Base class for ``sktime`` forecasters.

    References
    ----------
    .. [1] Tabby model card:
       https://huggingface.co/paris-noah/Tabby
    .. [2] TabbyTSFM repository:
       https://github.com/huawei-noah/trustworthyAI/tree/master/TabbyTSFM
    .. [3] Xie et al., "Tabby: An Open Pretraining Recipe for Time Series
       Foundation Models", https://arxiv.org/abs/2609.13956

    The model card and repository code are licensed under CC BY-NC 4.0; the
    weights are downloaded by Transformers at runtime and are not distributed
    by ``sktime``.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.tabby import TabbyForecaster  # doctest: +SKIP
    >>> y = load_airline()
    >>> forecaster = TabbyForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "category": ["deep_learning", "foundation_model"],
        "capability:multivariate": False,
        "capability:exogenous": False,
        "capability:missing_values": True,
        "capability:pred_int": True,
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "requires-fh-in-fit": False,
        "python_dependencies": ["torch", "transformers", "einops"],
        "authors": ["saptadeep12"],
        "maintainers": ["saptadeep12"],
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path: str = "paris-noah/Tabby",
        device: str = "auto",
        dtype=None,
        revision: str | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.revision = revision

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Store the observed series and load the pretrained model."""
        self.context_ = y.copy()
        self.model_ = self._load_model()
        return self

    def _predict(self, fh, X=None):
        """Return point forecasts from Tabby's median quantile."""
        predictions, relative_steps, index = self._run_forecast(fh)
        # Tabby's native grid is 1%-99%; index relative steps from one.
        values = predictions[0, 49, relative_steps - 1].astype("float64")

        if isinstance(self.context_, pd.Series):
            return pd.Series(values, index=index, name=self.context_.name)
        return pd.DataFrame(
            values, index=index, columns=self.context_.columns, dtype="float64"
        )

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Return requested Tabby quantiles, interpolating its native grid."""
        if alpha is None:
            alpha = [0.1, 0.5, 0.9]
        alpha = [float(value) for value in alpha]
        if any(value < 0.01 or value > 0.99 for value in alpha):
            raise ValueError(
                "Tabby quantiles are available only for alpha in [0.01, 0.99]."
            )

        predictions, relative_steps, index = self._run_forecast(fh)
        grid = np.arange(1, 100, dtype=np.float64) / 100
        # Map absolute requested steps back to the generated horizon.
        values = np.asarray(
            [
                [
                    self._interpolate_quantile(value, grid, predictions[0, :, step - 1])
                    for value in alpha
                ]
                for step in relative_steps
            ],
            dtype="float64",
        )
        var_name = self.context_.name
        if var_name is None:
            var_name = 0
        columns = pd.MultiIndex.from_product([[var_name], alpha])

        return pd.DataFrame(values, index=index, columns=columns)

    def _predict_proba(self, fh, X=None, marginal=True):
        """Return a histogram distribution interpolated from Tabby's quantiles."""
        from skpro.distributions import HistogramQPD

        predictions, relative_steps, index = self._run_forecast(fh)
        grid = np.arange(1, 100, dtype=np.float64) / 100
        values = predictions[0, :, relative_steps - 1]
        var_name = self.context_.name
        if var_name is None:
            var_name = 0
        # HistogramQPD preserves the native 99-quantile grid and its ppf uses
        # the same linear interpolation as _predict_quantiles.
        row_index = pd.MultiIndex.from_product([grid, index])
        quantiles = pd.DataFrame(
            values.T.reshape(-1, 1), index=row_index, columns=[var_name]
        )
        return HistogramQPD(quantiles, tails="mass", index=index, columns=[var_name])

    @staticmethod
    def _interpolate_quantile(alpha, grid, values):
        """Return the closest native quantile or interpolate between levels."""
        closest_idx = int(np.argmin(np.abs(grid - alpha)))
        if abs(grid[closest_idx] - alpha) <= 1e-8:
            return values[closest_idx]
        return np.interp(alpha, grid, values)

    def _run_forecast(self, fh):
        """Run Tabby once and return quantiles at all requested steps.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon, relative to the fitted series cutoff.

        Returns
        -------
        predictions : np.ndarray
            Quantile predictions with shape ``(1, 99, max(fh))``.
        relative_steps : np.ndarray
            Requested relative forecasting steps.
        index : pd.Index
            Absolute index corresponding to ``fh``.
        """
        import torch

        relative_fh = fh.to_relative(self.cutoff)
        relative_steps = np.asarray(relative_fh.to_numpy(), dtype=np.int64)
        horizon = int(relative_steps.max())

        model = self._load_model()
        config = getattr(model, "config", None)
        context_length = getattr(config, "context_length", 8192)
        patch_size = getattr(config, "patch_size", 16)
        max_horizon = context_length - patch_size
        # The model masks at least 128 future positions, but does not reject a
        # horizon that cannot fit after its 16-point patch.
        if max(horizon, 128) > max_horizon:
            raise ValueError(
                "Requested forecasting horizon exceeds Tabby's model capacity: "
                f"max requested step={horizon}, maximum supported step="
                f"{max_horizon}."
            )

        # The vendored model applies the 8,192-position context truncation.
        # NaNs are deliberately retained so its observed-input mask sees them.
        history = np.asarray(self.context_, dtype=np.float32).reshape(-1)
        history_tensor = torch.as_tensor(history, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            output = model.predict(history_tensor, prediction_length=horizon)

        predictions = output.detach().float().cpu().numpy()
        if predictions.shape != (1, 99, horizon):
            raise ValueError(
                "Tabby returned quantiles with unexpected shape "
                f"{predictions.shape}; expected {(1, 99, horizon)}."
            )

        return predictions, relative_steps, fh.to_absolute_index(self.cutoff)

    def _load_model(self):
        """Load the checkpoint or retrieve it from the shared model cache."""
        self.model_ = _CachedTabby(
            key=self._get_unique_key(),
            model_path=self.model_path,
            revision=self.revision,
            device=self.device,
            dtype=self.dtype,
        ).load()
        return self.model_

    def _get_unique_key(self):
        """Return the cache key for this model configuration."""
        return (
            self.model_path,
            self.revision,
            self.device,
            self.dtype,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        return [
            {},
            {"device": "cpu"},
        ]


@_multiton
class _CachedTabby:
    """Multiton-backed cache wrapper for a loaded Tabby model."""

    def __init__(
        self,
        key,
        model_path,
        revision,
        device,
        dtype,
    ):
        self.key = key
        self.model_path = model_path
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.model_ = None

    def load(self):
        """Load the model on the requested device once."""
        if self.model_ is not None:
            return self.model_

        import torch

        from sktime.libs.tabby import TabbyConfig, TabbyForForecasting

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        config = TabbyConfig.from_pretrained(self.model_path, revision=self.revision)
        if config.model_type != "tabby":
            raise ValueError(
                "Tabby checkpoint has incompatible config model_type: "
                f"{config.model_type!r}; expected 'tabby'."
            )

        load_kwargs = {
            "revision": self.revision,
            "config": config,
            "output_loading_info": True,
        }
        if self.dtype is not None:
            load_kwargs["dtype"] = self.dtype

        self.model_, loading_info = TabbyForForecasting.from_pretrained(
            self.model_path, **load_kwargs
        )
        missing_keys = loading_info.get("missing_keys", [])
        unexpected_keys = loading_info.get("unexpected_keys", [])
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Tabby checkpoint did not load exactly. "
                f"Missing keys: {missing_keys}; "
                f"unexpected keys: {unexpected_keys}."
            )
        self.model_.loading_info_ = loading_info
        self.model_ = self.model_.to(device)
        self.model_.eval()
        return self.model_
