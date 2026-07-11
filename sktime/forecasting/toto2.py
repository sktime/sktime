# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements ToTo2 forecaster."""

# This product includes software developed at Datadog, Copyright 2025 Datadog, Inc.

__author__ = ["siddharth7113"]
__all__ = ["Toto2Forecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class Toto2Forecaster(BaseForecaster):
    """Toto 2.0 foundation model forecaster for zero-shot forecasting.

    Direct interface to the forecaster from DataDog/toto [1]_.

    Toto 2.0 is the latest generation, featuring a u-μP-scaled transformer with
    alternating time/variate attention and quantile-based probabilistic
    forecasting. It supports zero-shot forecasting and can process multiple
    variables, generating both point forecasts and uncertainty estimates via a
    quantile head. It supports variable prediction horizons and context lengths.

    Parameters
    ----------
    model_path : str, optional (default="Datadog/Toto-2.0-22m")
        Path to the Toto 2.0 HuggingFace model. Available checkpoints include
        "Datadog/Toto-2.0-4m", "Datadog/Toto-2.0-22m", "Datadog/Toto-2.0-313m",
        "Datadog/Toto-2.0-1B", and "Datadog/Toto-2.0-2.5B".
    decode_block_size : int or None, optional (default=None)
        Decoding strategy. None (single forward pass) is faster and best for
        short horizons (used for all leaderboard results). A value such as 768
        (block decode) gives better long-term stability for horizons >~1000.
        If set, must be a multiple of the model's patch size.
    device : str or None, optional (default=None)
        Device on which to run the model ('cpu' or 'cuda'). If None, uses
        'cuda' when available, otherwise 'cpu'.
    seed : int or None, optional (default=None)
        Random seed for reproducibility; if None, a random seed is drawn.

    Notes
    -----
    Toto-2 emits forecasts at a fixed grid of quantile levels (0.1, 0.2, ..., 0.9).
    ``predict_proba`` returns a ``skpro`` ``HistogramQPD`` built from this grid: it
    interpolates linearly between adjacent grid quantiles, and (via ``tails="mass"``)
    clamps levels outside ``[0.1, 0.9]`` to the nearest grid quantile (e.g. a 0.05
    request returns the 0.1 quantile, so intervals wider than 80% coverage saturate).
    Interpolation assumes the grid quantiles are monotone in the level.

    References
    ----------
    .. [1] https://github.com/DataDog/toto

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.toto2 import Toto2Forecaster
    >>> y = load_airline()

    Zero-shot forecasting with the default model:

    >>> forecaster = Toto2Forecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Probabilistic forecasting. Toto-2 emits a fixed quantile grid (0.1, ..., 0.9);
    other levels come from a HistogramQPD (linear interpolation, clamped tails):

    >>> forecaster = Toto2Forecaster(
    ...     model_path="Datadog/Toto-2.0-4m"
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> intervals = forecaster.predict_interval(
    ...     fh=[1, 2, 3], coverage=0.9
    ... )  # doctest: +SKIP

    Long-horizon forecasting with block decoding. Block decoding only engages
    when the horizon spans multiple patches (i.e. exceeds ``decode_block_size``):

    >>> forecaster = Toto2Forecaster(
    ...     model_path="Datadog/Toto-2.0-22m", decode_block_size=768
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=list(range(1, 1001)))  # doctest: +SKIP
    """

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "None",
        "capability:multivariate": True,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:pretrain": False,
        # ownership and contribution tags
        "authors": ["siddharth7113"],
        "maintainers": ["siddharth7113"],
        "python_version": ">=3.12",
        "python_dependencies": ["toto-models", "skpro>=2.14"],
        # CI and test flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path: str = "Datadog/Toto-2.0-22m",
        decode_block_size: int | None = None,
        device: str | None = None,
        seed: int | None = None,
    ):
        self.model_path = model_path
        self.decode_block_size = decode_block_size
        self.device = device
        self.seed = seed

        super().__init__()

    def __post_init__(self):
        """Post-init constructor logic.

        Resolves the random seed used for reproducible sampling. No heavy
        compute happens here; model loading is deferred to ``_fit``/``_predict``.
        """
        self._seed = np.random.randint(0, 2**31) if self.seed is None else self.seed

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.

            * if self.get_tag("capability:multivariate")==False:
              guaranteed to be univariate (e.g., single-column for DataFrame)
            * if self.get_tag("capability:multivariate")==True: no restrictions apply,
              the method should handle uni- and multivariate y appropriately

        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        import torch

        if self.device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        values = y.values.T
        target = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(self._device)
        target_mask = torch.ones_like(target, dtype=torch.bool)
        n_var = target.shape[1]
        series_ids = torch.zeros(1, n_var, dtype=torch.long, device=self._device)
        self._inputs_ = {
            "target": target,
            "target_mask": target_mask,
            "series_ids": series_ids,
        }

        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        model, quantiles = self._run_forecast(fh)
        median_idx = model.output_head.knots.index(0.5)
        all_predictions = quantiles[median_idx].squeeze(0).cpu().numpy().T
        pred_index = fh.to_absolute(self._cutoff)._values
        relative_indices = fh.to_relative(self._cutoff) - 1
        selected = all_predictions[relative_indices]

        return pd.DataFrame(selected, index=pred_index, columns=self._y.columns)

    def _run_forecast(self, fh):
        """Load the cached model, align the context, and run the raw forecast.

        Shared helper for ``_predict`` and ``_predict_quantiles``.

        Returns
        -------
        model : Toto2Model
            the loaded model; exposes ``output_head.knots`` (the quantile grid)
        quantiles : torch.Tensor
            shape ``[n_quantiles, batch, n_var, horizon_padded]``
        """
        import torch

        torch.manual_seed(self._seed)
        model = _CachedToto2Forecaster(
            key=str((self.model_path, self._device)),
            model_path=self.model_path,
            device=self._device,
        ).load_from_checkpoint()

        patch_size = model.config.patch_size
        target, target_mask, has_missing = self._align_context(
            self._inputs_["target"], self._inputs_["target_mask"], patch_size
        )
        series_ids = self._inputs_["series_ids"]

        prediction_length = int(max(fh.to_relative(self._cutoff)))

        quantiles = model.forecast(
            {"target": target, "target_mask": target_mask, "series_ids": series_ids},
            horizon=prediction_length,
            decode_block_size=self.decode_block_size,
            has_missing_values=has_missing,
        )
        return model, quantiles

    @staticmethod
    def _align_context(target, target_mask, patch_size):
        """Left-pad the context so its length is a multiple of ``patch_size``.

        The model consumes the context in whole patches, so its length must be a
        multiple of ``patch_size``. We left-pad (never trim) so no observed point
        is discarded; padded positions are masked and handled natively by the model.

        Parameters
        ----------
        target : torch.Tensor, shape [batch, n_var, time]
        target_mask : torch.Tensor of bool, same shape as ``target``
        patch_size : int

        Returns
        -------
        target, target_mask : torch.Tensor
            padded so the time axis is a multiple of ``patch_size``
        has_missing : bool
            True if padding was added (enables the masked attention path);
            False if already aligned (enables the Flash-Attention path).
        """
        import torch

        remainder = target.shape[-1] % patch_size
        if remainder == 0:
            return target, target_mask, False

        pad = patch_size - remainder
        target = torch.nn.functional.pad(target, (pad, 0))
        pad_mask = torch.zeros(
            (*target_mask.shape[:-1], pad), dtype=torch.bool, device=target_mask.device
        )
        target_mask = torch.cat([pad_mask, target_mask], dim=-1)
        return target, target_mask, True

    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon (not optional)
            The forecasting horizon encoding the time stamps to forecast at.
            if has not been passed in fit, must be passed, not optional
        X : sktime time series object, optional (default=None)
                Exogeneous time series for the forecast
            Should be of same scitype (Series, Panel, or Hierarchical) as y in fit
            if self.get_tag("X-y-must-have-same-index"),
                X.index must contain fh.index and y.index both
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint
        """
        import numpy as np
        from skpro.distributions import HistogramQPD

        model, quantiles = self._run_forecast(fh)

        knots = model.output_head.knots
        q = quantiles.squeeze(1).cpu().numpy()

        var_names = self._y.columns
        pred_index = fh.to_absolute(self._cutoff)._values
        relative_indices = np.asarray(fh.to_relative(self._cutoff)) - 1

        row_index = pd.MultiIndex.from_product([knots, pred_index])
        selected = q[:, :, relative_indices]
        data = selected.transpose(0, 2, 1).reshape(len(knots) * len(pred_index), -1)
        q_df = pd.DataFrame(data, index=row_index, columns=var_names)

        return HistogramQPD(q_df, tails="mass", index=pred_index, columns=var_names)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [
            {"model_path": "Datadog/Toto-2.0-4m", "device": "cpu"},
            {
                "model_path": "Datadog/Toto-2.0-4m",
                "device": "cpu",
                "decode_block_size": 64,
            },
        ]


@_multiton
class _CachedToto2Forecaster:
    """Cached Toto-2 model.

    Toto-2 is a zero-shot, immutable model, so a single loaded instance can be
    shared across forecasters with the same configuration without side effects.
    The ``_multiton`` decorator keys instances by ``(model_path, device)`` so the
    (heavy) model is loaded at most once per configuration.
    """

    def __init__(self, key, model_path, device):
        self.key = key
        self.model_path = model_path
        self.device = device
        self.model = None

    def load_from_checkpoint(self):
        if self.model is not None:
            return self.model

        from toto2 import Toto2Model

        model = Toto2Model.from_pretrained(self.model_path)
        self.model = model.to(self.device).eval()
        return self.model
