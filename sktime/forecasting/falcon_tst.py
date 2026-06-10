# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Falcon-TST forecaster for ``sktime``."""

__author__ = ["geetu040"]

__all__ = ["FalconTSTForecaster"]

from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class FalconTSTForecaster(BaseForecaster):
    """Falcon-TST forecaster via Hugging Face ``transformers``."""

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pretrain": False,
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "python_dependencies": ["torch", "transformers", "einops"],
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path="ant-intl/Falcon-TST_Large",
        config=None,
        device_map="cpu",
        dtype=None,
        quantization_config=None,
        revin=True,
    ):
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.dtype = dtype
        self.quantization_config = quantization_config
        self.revin = revin

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        self.model_ = self._load_model()
        self.context_ = y

        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon."""
        import torch

        self.model_ = self._load_model()

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1
        forecast_horizon = np.max(preds_idx) + 1

        past_values = self.context_.to_numpy()
        is_univariate = past_values.shape[1] == 1
        if is_univariate:
            past_values = past_values[:, 0]

        past_values = np.expand_dims(past_values, axis=0)
        past_values = torch.from_numpy(past_values)
        past_values = past_values.to(self.model_.dtype)
        past_values = past_values.to(self.model_.device)

        output = self.model_.predict(
            past_values,
            forecast_horizon=forecast_horizon,
            revin=self.revin,
        )
        preds = output.detach().float().cpu().numpy()

        index = fh.to_absolute(self._cutoff)._values
        if is_univariate:
            preds = preds.squeeze(axis=0)
            preds = preds[preds_idx]
            name = self.context_.columns[0]
            return pd.Series(preds, index=index, name=name)

        preds = preds.squeeze(axis=0)
        preds = preds[preds_idx, :]
        return pd.DataFrame(preds, index=index, columns=self.context_.columns)

    def _load_model(self):
        """Load a Falcon-TST model instance."""
        if hasattr(self, "model_") and self.model_ is not None:
            return self.model_

        if self.model_path is not None:
            self.model_ = self._load_model_from_path()
        else:
            self.model_ = self._load_model_from_config()

        return self.model_

    def _load_model_from_path(self):
        """Load pretrained Falcon-TST weights from ``self.model_path``."""
        from sktime.libs.falcon_tst import FalconTSTForPrediction

        model_kwargs = {
            "device_map": self.device_map,
            "quantization_config": self.quantization_config,
        }
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype

        return FalconTSTForPrediction.from_pretrained(
            self.model_path,
            **model_kwargs,
        )

    def _load_model_from_config(self):
        """Initialize a Falcon-TST model from config without pretrained weights."""
        from sktime.libs.falcon_tst import FalconTSTConfig, FalconTSTForPrediction

        warn(
            "Initializing Falcon-TST from config creates random weights. "
            "Falcon-TST training is not supported by this estimator, so these "
            "weights will stay random and are only suitable for tests or local "
            "experimentation.",
            UserWarning,
            stacklevel=2,
        )

        config = deepcopy(self.config)
        if not config:
            config = FalconTSTConfig()
        if isinstance(config, dict):
            config = FalconTSTConfig.from_dict(config)

        model = FalconTSTForPrediction(config)
        model = model.to(self.device_map)
        if self.dtype is not None:
            model = model.to(dtype=self.dtype)

        return model
