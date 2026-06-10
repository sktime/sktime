# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Falcon-TST forecaster for ``sktime``."""

__author__ = ["geetu040"]

__all__ = ["FalconTSTForecaster"]

from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class FalconTSTForecaster(BaseForecaster):
    """Falcon-TST forecaster via Hugging Face ``transformers``.

    This forecaster wraps Falcon-TST prediction models [1]_, [2]_ from Hugging
    Face and exposes them through the ``sktime`` forecasting interface.
    ``fit`` loads the model and stores the observed series as forecasting
    context; it does not train or fine-tune Falcon-TST weights.

    Parameters
    ----------
    model_path : str, default="ant-intl/Falcon-TST_Large"
        Hugging Face repository identifier or local path to a Falcon-TST
        checkpoint. If ``None``, a model is created from ``config`` with random
        weights.
    config : FalconTSTConfig or dict, optional (default=None)
        Model configuration used when ``model_path=None``. If provided as a
        ``dict``, it is converted with ``FalconTSTConfig.from_dict``.
    device_map : str, dict, int, or torch.device, default="cpu"
        Device placement used for model loading or local initialization.
    dtype : torch.dtype or str, optional (default=None)
        Data type used for model loading, for example ``torch.float16``,
        ``torch.bfloat16``, or ``"auto"``.
    quantization_config : transformers.quantizers.HfQuantizer, optional
        Quantization configuration compatible with
        ``transformers.PreTrainedModel.from_pretrained``.
    revin : bool, default=True
        Whether to use RevIN normalization during Falcon-TST prediction.

    Notes
    -----
    - Falcon-TST training and fine-tuning are not supported by this estimator.
    - Falcon-TST supports multivariate targets, handled as channels.
    - Exogenous data and quantile prediction are not supported.
    - Loaded models are cached by model-loading inputs to avoid repeated model
      instantiation.

    References
    ----------
    .. [1] Falcon-TST repository:
       https://github.com/AntGroup/Falcon-TST
    .. [2] Falcon-TST model card:
       https://huggingface.co/ant-intl/Falcon-TST_Large

    Examples
    --------
    Univariate zero-shot forecasting:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.falcon_tst import FalconTSTForecaster
    >>> y = load_airline()
    >>> forecaster = FalconTSTForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Multivariate zero-shot forecasting:

    >>> from sktime.forecasting.falcon_tst import FalconTSTForecaster
    >>> y = y.to_frame("a").assign(b=lambda x: x["a"])  # doctest: +SKIP
    >>> forecaster = FalconTSTForecaster(revin=False)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Quantized loading:

    >>> import torch  # doctest: +SKIP
    >>> from transformers import BitsAndBytesConfig  # doctest: +SKIP
    >>> forecaster = FalconTSTForecaster(  # doctest: +SKIP
    ...     device_map="auto",
    ...     dtype=torch.bfloat16,
    ...     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    ... )
    """

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

        self.model_ = _CachedFalconTST(
            key=self._get_unique_key(),
            model_path=self.model_path,
            config=self.config,
            device_map=self.device_map,
            dtype=self.dtype,
            quantization_config=self.quantization_config,
        ).load()

        return self.model_

    def _get_unique_key(self):
        """Build cache key for the multiton model loader."""
        key = {
            "model_path": self.model_path,
            "config": self.config,
            "device_map": self.device_map,
            "dtype": self.dtype,
            "quantization_config": self.quantization_config,
        }
        return str(sorted(key.items()))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        config = {
            "num_hidden_layers": 1,
            "hidden_size": 4,
            "ffn_hidden_size": 8,
            "num_attention_heads": 1,
            "seq_length": 8,
            "shared_patch_size": 2,
            "patch_size_list": [4],
            "expert_num_layers": 1,
            "multi_forecast_head_list": [2],
            "autoregressive_step_list": [1],
            "num_experts": 1,
            "moe_router_topk": 1,
            "moe_ffn_hidden_size": 8,
            "moe_shared_expert_intermediate_size": 8,
            "use_cpu_initialization": True,
        }
        return [
            {
                "model_path": None,
                "config": config,
                "device_map": "cpu",
            },
            {
                "model_path": None,
                "config": config,
                "device_map": "cpu",
                "revin": False,
            },
        ]


@_multiton
class _CachedFalconTST:
    """Multiton-backed cache wrapper for a loaded Falcon-TST model."""

    def __init__(
        self,
        key,
        model_path,
        config,
        device_map,
        dtype,
        quantization_config,
    ):
        self.key = key
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.dtype = dtype
        self.quantization_config = quantization_config
        self.model_ = None

    def load(self):
        """Load model if needed and return cached instance."""
        if self.model_ is not None:
            return self.model_

        if self.model_path is not None:
            self.model_ = self._load_from_path()
        else:
            self.model_ = self._load_randomly()

        return self.model_

    def _load_from_path(self):
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

    def _load_randomly(self):
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
            dtype = _coerce_torch_dtype(self.dtype)
            if dtype is not None:
                model = model.to(dtype=dtype)

        return model


def _coerce_torch_dtype(dtype):
    """Coerce string dtype names to ``torch.dtype`` for local initialization."""
    if dtype == "auto":
        return None
    if isinstance(dtype, str):
        import torch

        dtype = dtype.removeprefix("torch.")
        return getattr(torch, dtype)
    return dtype
