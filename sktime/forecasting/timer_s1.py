# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Timer-S1 forecaster adapter for sktime."""

__author__ = ["geetu040"]
__all__ = ["TimerS1Forecaster"]

from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.singleton import _multiton


class TimerS1Forecaster(BaseForecaster):
    """Timer-S1 forecaster via Hugging Face transformers.

    Rough draft adapter for ByteDance Timer-S1.

    This currently supports:
    - zero-shot forecasting through fit + predict
    - quantile forecasts through predict_quantiles

    Notes
    -----
    Timer-S1 currently requires ``trust_remote_code=True`` because the modeling
    code is provided in the Hugging Face repository. This implementation keeps
    that inside the loader for now, but for sktime this should ideally point to
    a trusted fork/copy of the model code.
    """

    _tags = {
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pretrain": False,
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers[torch]~=4.57.1", "accelerate"],
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path="bytedance-research/Timer-S1",
        config=None,
        generate_kwargs=None,
        device_map=None,
        device="cpu",
        revin=True,
        peft_config=None,
    ):
        self.model_path = model_path
        self.config = config
        self.generate_kwargs = generate_kwargs
        self.device_map = device_map
        self.device = device
        self.revin = revin
        self.peft_config = peft_config

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster.

        For Timer-S1, fit only stores the context series and loads the model.
        """
        self.model_ = self._load_model()
        self.context_ = y
        return self

    def _predict(self, fh, X=None):
        """Return point forecasts.

        Timer-S1 returns quantile forecasts from generate. We use the median
        quantile, i.e. alpha=0.5, as the point forecast.
        """
        self.model_ = self._load_model()

        if fh is None:
            fh = self.fh

        fh_rel = fh.to_relative(self.cutoff)
        preds_idx = fh_rel._values.values - 1
        forecast_length = int(np.max(preds_idx) + 1)

        seqs = _series_to_tensor(self.context_, self.model_)

        generate_kwargs = {} if self.generate_kwargs is None else self.generate_kwargs
        output = self.model_.generate(
            seqs,
            max_new_tokens=forecast_length,
            revin=self.revin,
            **generate_kwargs,
        )

        # expected shape: batch_size x quantile_num x forecast_length
        if output.ndim != 3:
            raise ValueError(
                "Unexpected Timer-S1 generate output shape. Expected a tensor "
                "with shape (batch, quantiles, horizon), but got "
                f"{tuple(output.shape)}."
            )

        median_idx = _TIMER_S1_QUANTILES.index(0.5)
        preds = output[0, median_idx, preds_idx]

        preds = preds.detach().cpu().numpy()

        return pd.Series(
            preds,
            index=fh.to_absolute(self._cutoff)._values,
            name=self.context_.name,
        )

    def _predict_quantiles(self, fh, X, alpha):
        """Return quantile forecasts."""
        self.model_ = self._load_model()

        if fh is None:
            fh = self.fh

        if alpha is None:
            alpha = _TIMER_S1_QUANTILES

        alpha = [round(a, 3) for a in alpha]
        available_quantiles = [round(q, 3) for q in _TIMER_S1_QUANTILES]

        if not set(alpha).issubset(set(available_quantiles)):
            raise ValueError(
                "Requested quantiles are not available for Timer-S1. "
                f"requested={alpha}, available={available_quantiles}."
            )

        fh_rel = fh.to_relative(self.cutoff)
        preds_idx = fh_rel._values.values - 1
        forecast_length = int(np.max(preds_idx) + 1)

        seqs = _series_to_tensor(self.context_, self.model_)

        generate_kwargs = {} if self.generate_kwargs is None else self.generate_kwargs
        output = self.model_.generate(
            seqs,
            max_new_tokens=forecast_length,
            revin=self.revin,
            **generate_kwargs,
        )

        if output.ndim != 3:
            raise ValueError(
                "Unexpected Timer-S1 generate output shape. Expected a tensor "
                "with shape (batch, quantiles, horizon), but got "
                f"{tuple(output.shape)}."
            )

        quantile_idx = [available_quantiles.index(a) for a in alpha]

        # output: batch x quantile x horizon
        preds = output[0, quantile_idx, :]
        preds = preds[:, preds_idx]
        preds = preds.T
        preds = preds.detach().cpu().numpy()

        name = self.context_.name if self.context_.name is not None else 0
        columns = pd.MultiIndex.from_product([[name], alpha])

        return pd.DataFrame(
            data=preds,
            index=fh.to_absolute(self._cutoff)._values,
            columns=columns,
        )

    def _load_model(self):
        """Load or retrieve cached Timer-S1 model."""
        if hasattr(self, "model_") and self.model_ is not None:
            return self.model_

        self.model_ = _CachedTimerS1(
            key=self._get_unique_key(),
            model_path=self.model_path,
            config=self.config,
            device=self.device,
            device_map=self.device_map,
            peft_config=self.peft_config,
        ).load()

        return self.model_

    def _get_unique_key(self):
        """Build cache key."""
        key = {
            "model_path": self.model_path,
            "config": self.config,
            "device": self.device,
            "device_map": self.device_map,
            "peft_config": self.peft_config,
        }
        return str(sorted(key.items()))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test params.

        This is intentionally tiny/random-init style. It may need adjustment
        after checking the actual Timer-S1 config fields.
        """
        return {
            "model_path": None,
            "config": {
                "architectures": ["TimerForCausalLM"],
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_attention_heads": 4,
                "num_experts": 4,
                "num_experts_per_token": 2,
                "num_hidden_layers": 1,
                "num_mtp_tokens": 16,
            },
            "device": "cpu",
            "device_map": None,
        }


@_multiton
class _CachedTimerS1:
    """Cached Timer-S1 loader."""

    def __init__(
        self,
        key,
        model_path,
        config,
        device,
        device_map,
        peft_config,
    ):
        self.key = key
        self.model_path = model_path
        self.config = config
        self.device = device
        self.device_map = device_map
        self.peft_config = peft_config
        self.model_ = None

    def load(self):
        """Load Timer-S1 model."""
        if self.model_ is not None:
            return self.model_

        from transformers import AutoConfig, AutoModelForCausalLM

        config = self.config

        if self.model_path is not None:
            if config is None:
                config = AutoConfig.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
            elif isinstance(config, dict):
                base_config = AutoConfig.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
                for key, value in config.items():
                    setattr(base_config, key, value)
                config = base_config

            kwargs = {
                "config": config,
                "trust_remote_code": True,
            }
            if self.device_map is not None:
                kwargs["device_map"] = self.device_map

            self.model_ = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **kwargs,
            )
        else:
            if config is None:
                raise ValueError(
                    "If model_path=None, config must be provided for Timer-S1."
                )

            if isinstance(config, dict):
                # This part is uncertain because Timer-S1 config class is remote.
                # For draft PR this is acceptable, but should be checked against
                # the actual Timer-S1 configuration file.
                config = AutoConfig.for_model(
                    "timer",
                    trust_remote_code=True,
                    **config,
                )

            self.model_ = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
            )

        if self.device_map is None:
            self.model_ = self.model_.to(self.device)

        self.model_.eval()

        if self.peft_config is not None:
            self.model_ = self._wrap_with_peft()

        return self.model_

    def _wrap_with_peft(self):
        """Wrap model with PEFT."""
        _check_soft_dependencies("peft", severity="error")

        from peft import get_peft_model

        return get_peft_model(self.model_, deepcopy(self.peft_config))


_TIMER_S1_QUANTILES = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]


def _series_to_tensor(y, model):
    """Convert sktime y series to Timer-S1 input tensor."""
    import torch

    values = y.to_numpy()

    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values.reshape(1, -1)
    else:
        raise ValueError(
            "Timer-S1 draft adapter currently supports only univariate series. "
            f"Received values with shape={values.shape}."
        )

    seqs = torch.tensor(values, dtype=torch.float32)

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    return seqs.to(device)
