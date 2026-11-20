# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tabby forecaster for ``sktime``."""

__author__ = ["saptadeep12", "B1h14", "ShifengXIE"]

__all__ = ["TabbyForecaster"]

import numpy as np
import pandas as pd

from sktime.datatypes import update_data
from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class TabbyForecaster(BaseForecaster):
    """Tabby probabilistic forecaster via Hugging Face ``transformers``.

    Tabby is a univariate time series foundation model. This forecaster will
    wrap its 99 quantile forecasts in the ``sktime`` forecasting interface;
    multivariate series are handled by ``sktime``'s vectorization.

    .. note::

        The Tabby weights and vendored code in sktime.libs.tabby
        are licensed under CC BY-NC 4.0
        (non-commercial use only), copied from
        https://huggingface.co/paris-noah/Tabby at revision
        ``1b2381d7f1b0982aa1d5f09ce01b99eb4ee57e01``. The weights are not
        bundled with ``sktime``.

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
    license_accepted : bool, optional, default=False
        Whether the user has read and accepted the CC BY-NC 4.0 license terms
        for the Tabby weights and vendored code. Must be set to ``True`` to
        use ``TabbyForecaster``; otherwise ``fit`` raises.

    Notes
    -----
    - The forecast horizon is capped at 8,176 steps (the model's 8,192
      context minus its patch size of 16). Larger horizons raise an error.
    - The vendored model truncates history to its 8,192-position context and
      masks the requested future positions.
    - NaNs in the history are passed to the model as is; no imputation or
      scaling is applied.
    - Requested quantiles outside the model's native range are clamped to the
      outer native quantiles.

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
    >>> forecaster = TabbyForecaster(license_accepted=True)  # doctest: +SKIP
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
        "capability:update": True,
        "requires-fh-in-fit": False,
        "python_dependencies": ["torch", "transformers>=5.15.1"],
        "tests:libs": ["sktime.libs.tabby"],
        "authors": ["saptadeep12", "B1h14", "ShifengXIE"],
        "maintainers": ["saptadeep12"],
        "tests:vm": True,
        "tests:specific": ["sktime.forecasting.tests.test_tabby"],
    }

    def __init__(
        self,
        model_path: str = "paris-noah/Tabby",
        device: str = "auto",
        dtype=None,
        revision: str | None = None,
        license_accepted: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.license_accepted = license_accepted

        super().__init__()

    def _check_license(self):
        """Raise unless the user has accepted the Tabby license."""
        if not self.license_accepted:
            raise ValueError(
                "Tabby weights and vendored code in sktime.libs.tabby "
                "are licensed under CC BY-NC 4.0 "
                "(non-commercial use only). You must read and accept these "
                "terms to use the forecaster. The license is printed "
                "at sktime.libs.tabby.LICENSE. To confirm "
                "acceptance, set `license_accepted=True`."
            )

    def _fit(self, y, X=None, fh=None):
        """Store the observed series and load the pretrained model."""
        self._check_license()
        self.context_ = y.copy()
        self.model_ = self._load_model()
        return self

    def _predict(self, fh, X=None):
        """Return point forecasts from Tabby's median quantile."""
        predictions, relative_steps, index, levels = self._run_forecast(fh)
        median_idx = int(np.argmin(np.abs(levels - 0.5)))
        values = predictions[0, median_idx, relative_steps - 1].astype("float64")
        return pd.Series(values, index=index, name=self.context_.name)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Return requested Tabby quantiles, interpolating its native grid.

        Requested levels outside the native range are clamped to the outer
        native quantiles.
        """
        if alpha is None:
            alpha = [0.1, 0.5, 0.9]
        alpha = [float(value) for value in alpha]
        predictions, relative_steps, index, levels = self._run_forecast(fh)
        # Map absolute requested steps back to the generated horizon.
        values = np.asarray(
            [
                [
                    np.interp(value, levels, predictions[0, :, step - 1])
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

        predictions, relative_steps, index, levels = self._run_forecast(fh)
        values = predictions[0, :, relative_steps - 1]
        var_name = self.context_.name
        if var_name is None:
            var_name = 0
        # HistogramQPD preserves the model's native quantile grid and its ppf uses
        # the same linear interpolation as _predict_quantiles.
        row_index = pd.MultiIndex.from_product([levels, index])
        quantiles = pd.DataFrame(
            values.T.reshape(-1, 1), index=row_index, columns=[var_name]
        )
        return HistogramQPD(quantiles, tails="mass", index=index, columns=[var_name])

    def _run_forecast(self, fh):
        """Run Tabby once and return quantiles at all requested steps.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon, relative to the fitted series cutoff.

        Returns
        -------
        predictions : np.ndarray
            Quantile predictions with shape ``(1, n_quantiles, max(fh))``.
        relative_steps : np.ndarray
            Requested relative forecasting steps.
        index : pd.Index
            Absolute index corresponding to ``fh``.
        levels : np.ndarray
            Native quantile levels emitted by the model.
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
        levels = np.asarray(model.quantile_levels, dtype=np.float64)
        if predictions.shape != (1, len(levels), horizon):
            raise ValueError(
                "Tabby returned quantiles with unexpected shape "
                f"{predictions.shape}; expected {(1, len(levels), horizon)}."
            )

        return predictions, relative_steps, fh.to_absolute_index(self.cutoff), levels

    def _update(self, y, X=None, update_params=True):
        """Extend the stored history without refitting the model."""
        self.context_ = update_data(self.context_, y)
        return self

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
            {"license_accepted": True},
            {"device": "cpu", "license_accepted": True},
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
        }
        if self.dtype is not None:
            load_kwargs["dtype"] = self.dtype

        self.model_ = TabbyForForecasting.from_pretrained(
            self.model_path, **load_kwargs
        )
        self.model_ = self.model_.to(device)
        self.model_.eval()
        return self.model_
