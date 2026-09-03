# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TimesFM-3 forecaster for ``sktime``.

This module wraps the Google Research TimesFM 3.0 foundation model via the
``timesfm`` PyPI package (``timesfm3`` module). It supports:

- zero-shot univariate and multivariate forecasting
- native past-only and past-and-future dynamic covariates
- quantile prediction through :meth:`predict_quantiles`
"""

__author__ = ["jhalucky"]
__all__ = ["TimesFM3Forecaster"]

from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class TimesFM3Forecaster(BaseForecaster):
    """TimesFM-3 zero-shot forecaster with native multivariate and covariate support.

    TimesFM 3.0 is a decoder-only time series foundation model from Google Research
    that forecasts multiple related series in a single forward pass and accepts
    past-only and past-and-future dynamic covariates natively [1]_, [2]_.

    This forecaster wraps ``timesfm3.TimesFM3Evaluator`` and exposes the standard
    ``sktime`` forecasting interface.

    Parameters
    ----------
    checkpoint_path : str, default="google/timesfm-3.0-pytorch"
        Hugging Face repository identifier or local path to a TimesFM 3 checkpoint.
    per_core_batch_size : int, default=4
        Batch size used by the upstream evaluator during inference.
    device : str or None, default="cpu"
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
    context_length : int or None, default=None
        Maximum number of most-recent observations used as model context.
        If ``None``, the full training series is used (up to the upstream limit).
    past_only_covariate_cols : list of str or None, default=None
        Subset of ``X`` columns treated as past-only covariates (known only in
        history). All other ``X`` columns are treated as past-and-future
        covariates and require future values in :meth:`predict` via ``X``.
        If ``None``, columns absent from predict-time ``X`` are treated as
        past-only; columns present in both fit and predict ``X`` are treated as
        past-and-future.
    predict_kwargs : dict or None, default=None
        Extra keyword arguments forwarded to ``TimesFM3Evaluator.predict_batch``,
        for example ``use_symmetric_averaging``, ``make_positive``,
        ``sort_quantiles``, ``use_znorm``, or ``padding_mode``.
    ignore_deps : bool, default=False
        If ``True``, soft dependency checks are skipped.

    Notes
    -----
    TimesFM 3.0 pretrained weights are distributed under the TimesFM
    Non-Commercial License v1.0 and are restricted to non-commercial,
    non-production use.

    References
    ----------
    .. [1] https://github.com/google-research/timesfm
    .. [2] https://huggingface.co/google/timesfm-3.0-pytorch

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timesfm3 import TimesFM3Forecaster
    >>> y = load_airline()
    >>> forecaster = TimesFM3Forecaster(device="cpu")  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "authors": ["jhalucky"],
        "maintainers": ["jhalucky"],
        "python_dependencies": ["timesfm[torch]>=3.0.0"],
        "capability:multivariate": True,
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "tests:vm": True,
        "tests:specific": ["sktime.forecasting.tests.test_timesfm3"],
        "tests:skip_by_name": [
            "test_persistence_via_pickle",
            "test_save_estimators_to_file",
        ],
    }

    def __init__(
        self,
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=4,
        device="cpu",
        context_length=None,
        past_only_covariate_cols=None,
        predict_kwargs=None,
        ignore_deps=False,
    ):
        self.checkpoint_path = checkpoint_path
        self.per_core_batch_size = per_core_batch_size
        self.device = device
        self.context_length = context_length
        self.past_only_covariate_cols = past_only_covariate_cols
        self.predict_kwargs = predict_kwargs
        self.ignore_deps = ignore_deps

        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters."""
        if self.ignore_deps:
            self.set_tags(python_dependencies=[])

    def __getstate__(self):
        """Return state for pickling, excluding unpickleable evaluator."""
        state = self.__dict__.copy()
        if "evaluator_" in state:
            state["evaluator_"] = None
        return state

    def __setstate__(self, state):
        """Restore state; evaluator will be reloaded on next use."""
        self.__dict__.update(state)

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        context = self._coerce_y_to_dataframe(y)
        if self.context_length is not None and len(context) > self.context_length:
            context = context.iloc[-self.context_length :]

        self.context_ = context
        self._y_columns = list(context.columns)
        self._is_univariate = len(self._y_columns) == 1

        if X is not None:
            X = X.loc[context.index]
            self._validate_covariate_columns(X)

        self.evaluator_ = self._load_evaluator()
        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon."""
        fh, preds_idx, horizon = self._validate_predict_fh(fh)
        forecast, _ = self._run_forecast(fh, X, horizon, return_quantiles=False)
        index = fh.to_absolute(self._cutoff)._values
        if self._is_univariate:
            return pd.Series(forecast[preds_idx], index=index, name=self._y_columns[0])

        forecast = forecast[:, preds_idx].T
        return pd.DataFrame(forecast, index=index, columns=self._y_columns)

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast."""
        fh, preds_idx, horizon = self._validate_predict_fh(fh)
        _, quantiles = self._run_forecast(fh, X, horizon, return_quantiles=True)

        model_quantiles = self._get_model_quantiles()
        alpha = [round(float(a), 3) for a in alpha]
        model_quantiles_rounded = [round(float(q), 3) for q in model_quantiles]
        if not set(alpha).issubset(set(model_quantiles_rounded)):
            raise ValueError(
                "Requested quantiles are not all available in model config: "
                f"requested={alpha}, available={model_quantiles_rounded}."
            )
        quantile_idx = [model_quantiles_rounded.index(a) for a in alpha]

        index = fh.to_absolute(self._cutoff)._values
        columns = pd.MultiIndex.from_product([self._y_columns, alpha])

        if self._is_univariate:
            preds = quantiles[preds_idx][:, quantile_idx]
            preds = preds.reshape(len(preds_idx), len(alpha))
        else:
            preds = quantiles[:, preds_idx, :][:, :, quantile_idx]
            preds = np.transpose(preds, (1, 0, 2))
            preds = preds.reshape(len(preds_idx), len(self._y_columns) * len(alpha))

        return pd.DataFrame(preds, index=index, columns=columns)

    def _run_forecast(self, fh, X, horizon, return_quantiles):
        """Call upstream evaluator and return point and/or quantile forecasts."""
        self.evaluator_ = self._load_evaluator()

        target = self.context_.values.T.astype(np.float32)
        if self._is_univariate:
            target = target.ravel()

        past_only, past_future = self._prepare_covariates(X, horizon)

        predict_kwargs = {"return_quantiles": return_quantiles}
        if self.predict_kwargs:
            predict_kwargs.update(deepcopy(self.predict_kwargs))

        outputs = list(
            self.evaluator_.predict_batch(
                contexts=[target],
                horizon=horizon,
                past_only_covariates=[past_only],
                past_future_covariates=[past_future],
                **predict_kwargs,
            )
        )
        out = outputs[0]
        return out.forecast, out.quantiles

    def _prepare_covariates(self, X_future, horizon):
        """Split sktime ``X`` into TimesFM past-only and past-future arrays."""
        if self._X is None:
            if X_future is not None:
                raise ValueError(
                    "X was not provided in fit but is provided in predict. "
                    "Provide past covariate values in fit when using exogenous "
                    "variables."
                )
            return None, None

        X_past = self._X.loc[self.context_.index]
        past_only_cols, past_future_cols = self._resolve_covariate_columns(X_future)

        past_only = None
        if past_only_cols:
            past_only = X_past[past_only_cols].values.T.astype(np.float32)

        past_future = None
        if past_future_cols:
            if X_future is None:
                raise ValueError(
                    "Past-and-future covariates require future values in predict. "
                    f"Missing future values for columns: {past_future_cols}."
                )
            missing = set(past_future_cols) - set(X_future.columns)
            if missing:
                raise ValueError(
                    "Predict-time X is missing past-and-future covariate columns: "
                    f"{sorted(missing)}."
                )

            future = X_future[past_future_cols].iloc[:horizon]
            if len(future) < horizon:
                raise ValueError(
                    f"Future covariates must cover the full horizon (need {horizon} "
                    f"steps, got {len(future)})."
                )
            combined = np.concatenate(
                [X_past[past_future_cols].values.T, future.values.T],
                axis=1,
            )
            past_future = combined.astype(np.float32)

        return past_only, past_future

    def _resolve_covariate_columns(self, X_future):
        """Return past-only and past-and-future column names from fit-time X."""
        all_cols = list(self._X.columns)
        if self.past_only_covariate_cols is not None:
            past_only = list(self.past_only_covariate_cols)
            unknown = set(past_only) - set(all_cols)
            if unknown:
                raise ValueError(
                    "past_only_covariate_cols contains columns not present in X: "
                    f"{sorted(unknown)}."
                )
            past_future = [col for col in all_cols if col not in past_only]
            return past_only, past_future

        if X_future is None:
            return all_cols, []

        past_future = [col for col in all_cols if col in X_future.columns]
        past_only = [col for col in all_cols if col not in past_future]
        return past_only, past_future

    def _validate_covariate_columns(self, X):
        """Ensure explicit past-only columns exist in X."""
        if self.past_only_covariate_cols is None:
            return
        unknown = set(self.past_only_covariate_cols) - set(X.columns)
        if unknown:
            raise ValueError(
                "past_only_covariate_cols contains columns not present in X: "
                f"{sorted(unknown)}."
            )

    def _validate_predict_fh(self, fh):
        """Return relative fh indices and horizon length."""
        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1
        horizon = int(np.max(preds_idx) + 1)
        return fh, preds_idx, horizon

    def _load_evaluator(self):
        """Load or retrieve cached TimesFM 3 evaluator."""
        if hasattr(self, "evaluator_") and self.evaluator_ is not None:
            return self.evaluator_

        self.evaluator_ = _CachedTimesFM3(
            key=self._get_unique_key(),
            model_config_kwargs=self._get_model_config_kwargs(),
        ).load()
        return self.evaluator_

    def _get_model_config_kwargs(self):
        """Build kwargs for upstream ``ModelConfig``."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "per_core_batch_size": self.per_core_batch_size,
            "device": self.device,
        }

    def _get_unique_key(self):
        """Build multiton cache key."""
        return str(sorted(self._get_model_config_kwargs().items()))

    def _get_model_quantiles(self):
        """Return quantile levels configured on the loaded evaluator."""
        self.evaluator_ = self._load_evaluator()
        return list(self.evaluator_.config.quantiles)

    @staticmethod
    def _coerce_y_to_dataframe(y):
        """Convert inner target type to ``pd.DataFrame``."""
        if isinstance(y, pd.Series):
            return y.to_frame()
        return y

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        return [
            {
                "checkpoint_path": "google/timesfm-3.0-pytorch",
                "per_core_batch_size": 1,
                "device": "cpu",
                "predict_kwargs": {
                    "use_symmetric_averaging": False,
                    "make_positive": False,
                },
            },
            {
                "checkpoint_path": "google/timesfm-3.0-pytorch",
                "per_core_batch_size": 1,
                "device": "cpu",
                "context_length": 64,
                "predict_kwargs": {
                    "use_symmetric_averaging": True,
                    "make_positive": True,
                },
            },
        ]


@_multiton
class _CachedTimesFM3:
    """Multiton-backed cache wrapper for a loaded TimesFM 3 evaluator."""

    def __init__(self, key, model_config_kwargs):
        self.key = key
        self.model_config_kwargs = model_config_kwargs
        self.evaluator_ = None

    def load(self):
        """Load evaluator if needed and return cached instance."""
        if self.evaluator_ is not None:
            return self.evaluator_

        from timesfm3 import ModelConfig, TimesFM3Evaluator

        config = ModelConfig(**self.model_config_kwargs)
        self.evaluator_ = TimesFM3Evaluator(config)
        return self.evaluator_
