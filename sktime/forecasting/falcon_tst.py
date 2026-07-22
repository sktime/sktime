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
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class FalconTSTForecaster(BaseForecaster):
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
    - Loaded models are cached via a multiton helper keyed by model-loading
      inputs to avoid repeated model instantiation.
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
        Possible keys:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        ffn_hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the feed-forward networks in the transformer layers.
        seq_length (`int`, *optional*, defaults to 2880):
            Maximum sequence length that the model can handle.
        add_bias_linear (`bool`, *optional*, defaults to `False`):
            Whether to add bias in linear layers.
        rope_theta (`int`, *optional*, defaults to 10000):
            The base period of the RoPE embeddings.
        num_hidden_layers (`int`, *optional*, defaults to 3):
            Number of hidden layers in the transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the
            transformer encoder.
        mask_pad_value (`float`, *optional*, defaults to 255.0):
            Value used for padding/masking in input sequences.
        expert_num_layers (`int`, *optional*, defaults to 4):
            Number of transformer layers within each expert.
        shared_patch_size (`int`, *optional*, defaults to 64):
            Size of patches for the shared expert.
        patch_size_list (`List[int]`, *optional*, defaults to [96, 64, 48, 24]):
            List of patch sizes for different experts.
        multi_forecast_head_list (`List[int]`, *optional*, defaults to [24, 96, 336]):
            List of forecast lengths for multi-head prediction.
        is_revin (`bool`, *optional*, defaults to `True`):
            Whether to use RevIN (Reversible Instance Normalization).
        params_dtype (`str`, *optional*, defaults to "bfloat16"):
            Data type for model parameters.
        use_cpu_initialization (`bool`, *optional*, defaults to `False`):
            Whether to initialize model parameters on CPU.
        rotary_interleaved (`bool`, *optional*, defaults to `False`):
            Whether to use interleaved rotary position embeddings.
        do_expert_forecast (`bool`, *optional*, defaults to `True`):
            Whether experts perform forecasting.
        residual_backcast (`bool`, *optional*, defaults to `True`):
            Whether to use residual connections for backcast.
        do_base_forecast (`bool`, *optional*, defaults to `False`):
            Whether to use base forecasting.
        heterogeneous_moe_layer (`bool`, *optional*, defaults to `True`):
            Whether to use heterogeneous MoE layers.
        test_data_seq_len (`int`, *optional*, defaults to 2880):
            Sequence length for test data.
        test_data_test_len (`int`, *optional*, defaults to 720):
            Test length for test data.
        autoregressive_step_list (`List[int]`, *optional*, defaults to [2, 4, 1]):
            List of autoregressive steps for different forecast heads.
        multi_forecast_head_type (`str`, *optional*, defaults to "single"):
            Type of multi-forecast head aggregation.
        num_experts (`int`, *optional*, defaults to 4):
            Number of experts in the MoE layer.
        moe_router_topk (`int`, *optional*, defaults to 2):
            Number of top experts to route each token to.
        moe_ffn_hidden_size (`int`, *optional*, defaults to 4096):
            Hidden size for MoE feed-forward networks.
        moe_shared_expert_intermediate_size (`int`, *optional*, defaults to 4096):
            Intermediate size for shared experts.
        init_method_std (`float`, *optional*, defaults to 0.06):
            Standard deviation for weight initialization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Range for weight initialization.
        moe_router_enable_expert_bias (`bool`, *optional*, defaults to `False`):
            Whether to enable expert bias in routing.
        moe_expert_final_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization at the end of each expert.
        transformer_input_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization to transformer inputs.
        moe_router_pre_softmax (`bool`, *optional*, defaults to `True`):
            Whether to apply softmax before routing.
        q_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization to query vectors.
        k_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization to key vectors.
        moe_router_score_function (`str`, *optional*, defaults to "softmax"):
            Score function for MoE routing ("softmax" or "sigmoid").
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embeddings.

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

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Load the model and store history for zero-shot forecasting.

        private _fit containing the core logic, called from fit

        This method does not train or fine-tune Falcon-TST weights.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.

            * if self.get_tag("capability:multivariate")==False:
              guaranteed to be univariate (e.g., single-column for DataFrame)
            * if self.get_tag("capability:multivariate")==True: no restrictions
              apply, the method should handle uni- and multivariate y
              appropriately

        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")
            == True. Otherwise, if not passed in _fit, guaranteed to be passed
            in _predict.
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self.model_ = self._load_model()
        self.context_ = y

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
            If not passed in _fit, guaranteed to be passed here.
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast.

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype"
            tag. Point predictions.
        """
        import torch

        self.model_ = self._load_model()

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1
        forecast_horizon = np.max(preds_idx) + 1

        past_values = self.context_.to_numpy()
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
        preds = preds.squeeze(axis=0)
        preds = preds[preds_idx, :]
        preds = pd.DataFrame(
            preds,
            index=fh.to_absolute(self._cutoff)._values,
            columns=self.context_.columns,
        )

        return preds

    def _load_model(self):
        """Load or retrieve a cached Falcon-TST model instance.

        Returns
        -------
        model : transformers.PreTrainedModel
            Loaded model according to ``self.device_map``. If ``self.model_``
            already exists, it is returned directly.
        """
        if hasattr(self, "model_") and self.model_ is not None:
            return self.model_

        self.model_ = _CachedFalconTST(
            key=self._get_unique_key(),
            model_path=self.model_path,
            config=self.config,
            device_map=self.device_map,
            quantization_config=self.quantization_config,
        ).load()

        return self.model_

    def _get_unique_key(self):
        """Build cache key for the multiton model loader.

        Returns
        -------
        key : str
            Deterministic string representation of model-loading attributes used
            by :class:`_CachedFalconTST`.
        """
        key = {
            "model_path": self.model_path,
            "config": self.config,
            "device_map": self.device_map,
            "quantization_config": self.quantization_config,
        }
        return str(sorted(key.items()))

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


@_multiton
class _CachedFalconTST:
    """Multiton-backed cache wrapper for a loaded Falcon-TST model.

    Instances are keyed externally (via :class:`FalconTSTForecaster`) so that
    repeated forecasters with equivalent loading parameters can share a single
    model instance in memory.

    Parameters
    ----------
    key : str
        Multiton key (stored for traceability).
    model_path : str or None
        Model identifier/path for ``from_pretrained``. If ``None``, create model
        from config only.
    config : transformers.PretrainedConfig or dict or None
        Configuration used for model loading/creation.
    device_map : str, dict, int, or torch.device
        Device placement for loading models.
    quantization_config : transformers.quantizers.HfQuantizer or None
        Quantization configuration used for model loading.
    """

    def __init__(
        self,
        key,
        model_path,
        config,
        device_map,
        quantization_config,
    ):
        self.key = key
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.quantization_config = quantization_config
        self.model_ = None

    def load(self):
        """Load model if needed and return cached instance.

        Returns
        -------
        model : transformers.PreTrainedModel
            Loaded Falcon-TST prediction model according to ``self.device_map``,
            and ``self.quantization_config``.

        Notes
        -----
        - If ``model_path`` is set, loads weights via ``from_pretrained``.
        - If ``model_path`` is ``None``, initializes from config with random
          weights. These weights are not trained by the estimator.
        """
        if self.model_ is not None:
            return self.model_

        if self.model_path is not None:
            self.model_ = self._load_from_path()
        else:
            self.model_ = self._load_randomly()

        return self.model_

    def _load_from_path(self):
        """Load Falcon-TST model weights from ``self.model_path``."""
        from sktime.libs.falcon_tst import FalconTSTForPrediction

        kwargs = {}
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        if self.quantization_config is not None:
            kwargs["quantization_config"] = self.quantization_config

        model = FalconTSTForPrediction.from_pretrained(self.model_path, **kwargs)

        return model

    def _load_randomly(self):
        """Initialize a Falcon-TST model randomly from config."""
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

        return model
