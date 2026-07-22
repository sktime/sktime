# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Falcon-TST forecaster for ``sktime``.

This module provides an ``sktime`` forecaster wrapping the local Falcon-TST
``transformers`` model implementation. It supports:

- zero-shot prediction through :meth:`fit` + :meth:`predict`
- univariate and multivariate target forecasting

Model training and fine-tuning are not supported. Calling :meth:`fit` only
loads the model and stores the observed series as forecasting context.
"""

__author__ = ["Harryx2019", "figolyd", "geetu040"]
# Hongjie Xia (Harryx2019), Yiding Liu (figolyd) for ant-intl/Falcon-TST

__all__ = ["FalconTSTForecaster"]

from copy import deepcopy
from warnings import warn

import numpy as np

from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    FoundationModelSpec,
    ModelHandle,
)


class FalconTSTForecaster(BaseFoundationForecaster):
    """Falcon-TST forecaster via Hugging Face ``transformers``.

    This forecaster wraps Falcon-TST prediction models [1]_, [2]_ from Hugging
    Face and exposes them through the ``sktime`` forecasting interface.

    The primary workflow is ``fit`` for zero-shot inference setup, which loads
    the model and stores history. It does not train or fine-tune model weights.
    Passing ``model_path=None`` initializes a Falcon-TST model from ``config``
    instead of loading pretrained weights. These random weights cannot be
    trained through this estimator and are mainly useful for tests or local
    experimentation.

    Notes
    -----
    - Falcon-TST training is not supported. The estimator performs only
      zero-shot forecasting from a loaded or randomly initialized model.
    - Falcon-TST supports multivariate targets, handled as independent
      channels by the model.
    - Exogenous data and quantile prediction are not supported.
    - Loaded models are shared through the foundation-model cache, keyed by
      model-loading inputs to avoid repeated model instantiation.
    - For reduced-memory loading, use ``quantization_config`` or a
      pre-quantized checkpoint.

    Parameters
    ----------
    model_path : str, default="ant-intl/Falcon-TST_Large"
        Hugging Face repository identifier or local path to a Falcon-TST
        checkpoint. If ``None``, a model is created from ``config``.
    config : FalconTSTConfig or dict, optional (default=None)
        Model configuration used when ``model_path=None``. If provided as a
        ``dict``, it is converted with ``FalconTSTConfig.from_dict``. If
        ``None`` and ``model_path=None``, the default ``FalconTSTConfig`` is
        used. This path creates random weights; the estimator does not provide
        training for those weights.
    device_map : str, dict, int, or torch.device, default="cpu"
        Device placement following the ``transformers`` ``device_map`` naming
        convention, for example ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or
        ``"auto"``.
    quantization_config : transformers.quantizers.HfQuantizer, optional
        Valid quantization configuration object compatible with
        ``transformers.PreTrainedModel.from_pretrained`` [3]_.
    revin : bool, default=True
        Whether to use RevIN normalization during Falcon-TST prediction.

    References
    ----------
    .. [1] Falcon-TST repository:
       https://github.com/AntGroup/Falcon-TST
    .. [2] Falcon-TST model card:
       https://huggingface.co/ant-intl/Falcon-TST_Large
    .. [3] Quantization docs:
       https://huggingface.co/docs/transformers/en/main_classes/quantization

    Examples
    --------
    Simple zero-shot forecasting with the default Falcon-TST checkpoint:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.falcon_tst import FalconTSTForecaster
    >>> y = load_airline()
    >>> # By default, loads ant-intl/Falcon-TST_Large.
    >>> forecaster = FalconTSTForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Reduced-memory inference with device placement and quantization:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.falcon_tst import FalconTSTForecaster
    >>> from transformers import BitsAndBytesConfig  # doctest: +SKIP
    >>> y = load_airline()
    >>> forecaster = FalconTSTForecaster(  # doctest: +SKIP
    ...     model_path="ant-intl/Falcon-TST_Large",
    ...     device_map="auto",
    ...     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Randomly initialized local model, useful for tests or local experimentation.
    This model is not trained by ``fit``; the weights stay random and should not
    be used as a trained forecaster:

    >>> from sktime.forecasting.falcon_tst import FalconTSTForecaster
    >>> forecaster = FalconTSTForecaster(  # doctest: +SKIP
    ...     model_path=None,
    ...     config={
    ...         "num_hidden_layers": 1,
    ...         "hidden_size": 4,
    ...         "ffn_hidden_size": 8,
    ...         "num_attention_heads": 1,
    ...         "seq_length": 8,
    ...         "shared_patch_size": 2,
    ...         "patch_size_list": [4],
    ...         "transformer_input_layernorm": True,
    ...         "expert_num_layers": 1,
    ...         "multi_forecast_head_list": [2],
    ...         "autoregressive_step_list": [1],
    ...         "num_experts": 1,
    ...         "moe_router_topk": 1,
    ...         "moe_ffn_hidden_size": 8,
    ...         "moe_shared_expert_intermediate_size": 8,
    ...         "use_cpu_initialization": True,
    ...     },
    ... )
    """

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "authors": ["Harryx2019", "figolyd", "geetu040"],
        # Hongjie Xia (Harryx2019), Yiding Liu (figolyd) for ant-intl/Falcon-TST
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers[torch]>=4.23.0,<5.0.0"],
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path="ant-intl/Falcon-TST_Large",
        config=None,
        device_map="cpu",
        quantization_config=None,
        revin=True,
    ):
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.quantization_config = quantization_config
        self.revin = revin
        model_spec = FoundationModelSpec(
            model_path=model_path,
            config=config if model_path is None else None,
            device=device_map,
            quantization_config=quantization_config,
            predict_extra_kwargs={"revin": revin},
        )
        super().__init__(model_spec=model_spec)

    def _resolve_config(self, config):
        """Copy dict and ``PretrainedConfig`` inputs without mutating them."""
        return None if config is None else deepcopy(config)

    def _load_model(self):
        """Load pretrained or config-only Falcon-TST into a model handle."""
        from sktime.libs.falcon_tst import FalconTSTConfig, FalconTSTForPrediction

        model_spec = self.model_spec_
        if model_spec.model_path is not None:
            pretrained_kwargs = {}
            if model_spec.device is not None:
                pretrained_kwargs["device_map"] = model_spec.device
            if model_spec.quantization_config is not None:
                pretrained_kwargs["quantization_config"] = (
                    model_spec.quantization_config
                )
            model = FalconTSTForPrediction.from_pretrained(
                model_spec.model_path,
                **pretrained_kwargs,
                **model_spec.load_extra_kwargs,
            )
        else:
            warn(
                "Initializing Falcon-TST from config creates random weights. "
                "Falcon-TST training is not supported by this estimator, so these "
                "weights will stay random and are only suitable for tests or local "
                "experimentation.",
                UserWarning,
                stacklevel=2,
            )

            model_config = deepcopy(model_spec.config)
            if not model_config:
                model_config = FalconTSTConfig()
            if isinstance(model_config, dict):
                model_config = FalconTSTConfig.from_dict(model_config)

            model = FalconTSTForPrediction(model_config)
            model = model.to(model_spec.device)

        return ModelHandle(model=model)

    def _inference(
        self,
        handle,
        context_y,
        context_X,
        future_X,
        pred_len,
        fh,
        alpha=None,
    ):
        """Run Falcon-TST prediction and return the full native horizon."""
        import torch

        model = handle.model
        predict_kwargs = self.model_spec_.predict_extra_kwargs
        past_values = torch.from_numpy(np.expand_dims(context_y.to_numpy(), axis=0))
        past_values = past_values.to(model.dtype).to(model.device)

        output = model.predict(
            past_values,
            forecast_horizon=pred_len,
            **predict_kwargs,
        )
        predictions = output.detach().float().cpu().numpy().squeeze(axis=0)
        return ForecastResult(mean=predictions)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If
            no special parameters are defined for a value, will return
            `"default"` set. There are currently no reserved values for
            forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class. Each dict is a
            parameter set to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test
            instance. `create_test_instance` uses the first (or only)
            dictionary in `params`.
        """
        config = {
            "num_hidden_layers": 1,
            "hidden_size": 4,
            "ffn_hidden_size": 8,
            "num_attention_heads": 1,
            "seq_length": 8,
            "shared_patch_size": 2,
            "patch_size_list": [4],
            "transformer_input_layernorm": True,
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
