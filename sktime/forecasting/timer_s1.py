# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Timer-S1 forecaster for ``sktime``.

This module provides an ``sktime`` forecaster wrapping the local Timer-S1
``transformers`` model implementation. It supports:

- zero-shot prediction through :meth:`fit` + :meth:`predict`
- quantile prediction through :meth:`predict_quantiles`

Model training and fine-tuning are not supported at the moment, though they may
be added in future. Calling :meth:`fit` only loads the model and stores the
observed series as forecasting context.
"""

__author__ = ["WenWeiTHU", "geetu040"]
# WenWeiTHU for bytedance-research/Timer-S1

__all__ = ["TimerS1Forecaster"]

from copy import deepcopy
from warnings import warn

import numpy as np

from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    FoundationModelSpec,
    ModelHandle,
)


class TimerS1Forecaster(BaseFoundationForecaster):
    """Timer-S1 forecaster via Hugging Face ``transformers``.

    This forecaster wraps Timer-S1 prediction models [1]_, [2]_ from Hugging
    Face and exposes them through the ``sktime`` forecasting interface.

    The primary workflow is ``fit`` for zero-shot inference setup, which loads
    the model and stores history. It does not train or fine-tune model weights.
    Passing ``model_path=None`` initializes a Timer-S1 model from ``config``
    instead of loading pretrained weights. These random weights cannot be
    trained through this estimator at the moment and are mainly useful for tests
    or local experimentation.

    Notes
    -----
    - Timer-S1 training is currently not supported, but may be added in future.
      The estimator performs only zero-shot forecasting from a loaded or
      randomly initialized model.
    - Quantile prediction is only available for quantiles present in
      ``model.config.quantiles``.
    - Loaded models are shared through the foundation-model cache, keyed by
      model-loading inputs to avoid repeated model instantiation.
    - The default Timer-S1 checkpoint has 8 billion parameters. For most
      hardware, reduced-memory loading with ``dtype`` and ``quantization_config``,
      or a pre-quantized checkpoint, is recommended.

    Parameters
    ----------
    model_path : str, default="bytedance-research/Timer-S1"
        Hugging Face repository identifier or local path to a Timer-S1 checkpoint.
        If ``None``, a model is created from ``config``.
    config : TimerS1Config or dict, optional (default=None)
        Model configuration used when ``model_path=None``. If provided as a
        ``dict``, it is converted with ``TimerS1Config.from_dict``. If ``None``
        and ``model_path=None``, the default ``TimerS1Config`` is used. This
        path creates random weights; the estimator does not currently provide
        training for those weights.
    device_map : str, dict, int, or torch.device, default="cpu"
        Device placement following the ``transformers`` ``device_map`` naming
        convention, for example ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or
        ``"auto"``.
    dtype : torch.dtype or str, optional (default=None)
        Data type used for model loading, following the ``transformers``
        ``dtype`` convention, for example ``torch.float16``,
        ``torch.bfloat16``, or ``"auto"``.
    quantization_config : transformers.quantizers.HfQuantizer, optional
        Valid quantization configuration object compatible with
        ``transformers.PreTrainedModel.from_pretrained`` [3]_.
    forward_kwargs : dict, optional (default=None)
        Additional keyword arguments forwarded to ``model.generate(...)`` during
        :meth:`predict` and :meth:`predict_quantiles`.
    deterministic : bool, default=False
        Whether point predictions should reset the ``transformers`` random seed
        before generation. Currently this is applied in predict methods.

    References
    ----------
    .. [1] Liu, Y., Su, X., Wang, S., Zhang, H., Liu, H., Wang, Y.,
       Ye, Z., Xiang, Y., Wang, J., and Long, M. (2026).
       Timer-S1: A Billion-Scale Time Series Foundation Model with Serial
       Scaling. arXiv. https://arxiv.org/abs/2603.04791
    .. [2] Timer-S1 model card:
       https://huggingface.co/bytedance-research/Timer-S1
    .. [3] Quantization docs:
       https://huggingface.co/docs/transformers/en/main_classes/quantization

    Examples
    --------
    Simple zero-shot forecasting with the default Timer-S1 checkpoint:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timer_s1 import TimerS1Forecaster
    >>> y = load_airline()
    >>> # By default, loads bytedance-research/Timer-S1.
    >>> forecaster = TimerS1Forecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Reduced-memory inference for the 8-billion-parameter model:

    >>> import torch  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timer_s1 import TimerS1Forecaster
    >>> from transformers import BitsAndBytesConfig  # doctest: +SKIP
    >>> y = load_airline()
    >>> forecaster = TimerS1Forecaster(  # doctest: +SKIP
    ...     model_path="bytedance-research/Timer-S1",
    ...     forward_kwargs={"revin": True},
    ...     device_map="auto",
    ...     dtype=torch.bfloat16,
    ...     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Loading a quantized smaller model directly:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timer_s1 import TimerS1Forecaster
    >>> y = load_airline()
    >>> forecaster = TimerS1Forecaster(  # doctest: +SKIP
    ...     model_path="geetu040/Timer-S1-quantized-4bit",
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Randomly initialized local model, useful for tests or local experimentation.
    This model is not trained by ``fit``; the weights stay random and should not
    be used as a trained forecaster:

    >>> from sktime.forecasting.timer_s1 import TimerS1Forecaster
    >>> forecaster = TimerS1Forecaster(  # doctest: +SKIP
    ...     model_path=None,
    ...     config={
    ...         "hidden_size": 16,
    ...         "intermediate_size": 16,
    ...         "num_attention_heads": 4,
    ...         "num_experts": 4,
    ...         "num_hidden_layers": 1,
    ...         "num_mtp_tokens": 1,
    ...     },
    ...     deterministic=True,
    ... )

    Quantile prediction:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timer_s1 import TimerS1Forecaster
    >>> y = load_airline()
    >>> forecaster = TimerS1Forecaster(  # doctest: +SKIP
    ...     model_path="geetu040/Timer-S1-quantized-4bit",
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict_quantiles(  # doctest: +SKIP
    ...     fh=[1, 2, 3],
    ...     alpha=[0.1, 0.5, 0.9],
    ... )
    """

    _tags = {
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "authors": ["WenWeiTHU", "geetu040"],
        # WenWeiTHU for bytedance-research/Timer-S1
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers[torch]>4.57.0,<5.0.0"],
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path="bytedance-research/Timer-S1",
        config=None,
        device_map="cpu",
        dtype=None,
        quantization_config=None,
        forward_kwargs=None,
        deterministic=False,
    ):
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.dtype = dtype
        self.quantization_config = quantization_config
        self.forward_kwargs = forward_kwargs
        self.deterministic = deterministic
        model_spec = FoundationModelSpec(
            model_path=model_path,
            config=config,
            device=device_map,
            dtype=dtype,
            quantization_config=quantization_config,
            random_state=42 if deterministic else None,
            load_extra_kwargs={"trust_remote_code": True},
            predict_extra_kwargs=forward_kwargs or {},
        )
        super().__init__(model_spec=model_spec)

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
        """Generate Timer-S1 quantiles and normalize the backend output."""
        import torch

        model = handle.model
        predict_kwargs = self.model_spec_.predict_extra_kwargs
        model_quantiles = [round(value, 3) for value in model.config.quantiles]
        requested = None if alpha is None else list(alpha)
        requested_rounded = (
            None if requested is None else [round(value, 3) for value in requested]
        )
        if requested_rounded is not None and not set(requested_rounded).issubset(
            model_quantiles
        ):
            raise ValueError(
                "Requested quantiles are not all available in model config: "
                f"requested={requested_rounded}, available={model_quantiles}."
            )

        quantiles = np.asarray(model.config.quantiles)
        weights = (quantiles[:-1] + quantiles[1:]) / 2
        weights = np.concatenate([[0.0], weights, [1.0]])
        weights = np.diff(weights)

        past_values = torch.from_numpy(context_y.iloc[:, 0].to_numpy()[None, :])
        past_values = past_values.to(dtype=model.dtype, device=model.device)

        output = model.generate(
            past_values,
            max_new_tokens=pred_len,
            **predict_kwargs,
        )

        quantile_values = output.squeeze(axis=0).detach().float().cpu().numpy()
        point_values = np.average(quantile_values, weights=weights, axis=0)
        result_quantiles = None
        if requested is not None:
            result_quantiles = {
                value: quantile_values[model_quantiles.index(rounded)].reshape(-1, 1)
                for value, rounded in zip(requested, requested_rounded)
            }

        return ForecastResult(
            mean=point_values.reshape(-1, 1),
            quantiles=result_quantiles,
        )

    def _load_model(self):
        """Load Timer-S1 and return its shared foundation-model handle."""
        if self.model_spec_.model_path is not None:
            model = self._load_from_path()
        else:
            model = self._load_randomly()
        return ModelHandle(model=model)

    def _load_from_path(self):
        """Load pretrained Timer-S1 weights."""
        from sktime.libs.timer_s1 import TimerS1ForPrediction

        model_spec = self.model_spec_
        return TimerS1ForPrediction.from_pretrained(
            model_spec.model_path,
            device_map=model_spec.device,
            dtype=model_spec.dtype,
            quantization_config=model_spec.quantization_config,
            **model_spec.load_extra_kwargs,
        )

    def _load_randomly(self):
        """Initialize Timer-S1 from a local configuration."""
        from sktime.libs.timer_s1 import TimerS1Config, TimerS1ForPrediction

        warn(
            "Initializing Timer-S1 from config creates random weights. "
            "Timer-S1 training is not supported by this estimator at the "
            "moment, so these weights will stay random and are only suitable "
            "for tests or local experimentation.",
            UserWarning,
            stacklevel=2,
        )

        model_spec = self.model_spec_
        model_config = deepcopy(model_spec.config)
        if not model_config:
            model_config = TimerS1Config()
        if isinstance(model_config, dict):
            model_config = TimerS1Config.from_dict(model_config)

        model = TimerS1ForPrediction(model_config).to(model_spec.device)
        if model_spec.dtype is not None:
            model = model.to(dtype=model_spec.dtype)
        return model

    def _get_unique_model_key(self):
        """Build a hashable key from all Timer-S1 loading inputs."""
        spec = self.model_spec_
        key_items = {
            "class": self.__class__.__name__,
            "model_path": spec.model_path,
            "config": spec.config if spec.model_path is None else None,
            "device_map": spec.device,
            "dtype": spec.dtype,
            "quantization_config": spec.quantization_config,
            "load_extra_kwargs": spec.load_extra_kwargs,
        }
        return self.__class__.__name__, str(sorted(key_items.items()))

    def _resolve_config(self, config):
        """Copy dict and ``TimerS1Config`` inputs without mutating either."""
        if config is None:
            return {}
        return deepcopy(config)

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
        intervals = [0.9, 0.75, 0.5, 0.25, 0.1, 0.05]
        quantiles = sorted(q for p in intervals for q in ((1 - p) / 2, (1 + p) / 2))
        quantiles = [0.1] + quantiles

        test_params = []

        test_param_1 = {
            "model_path": None,
            "config": {
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_attention_heads": 4,
                "num_experts": 4,
                "num_hidden_layers": 1,
                "num_mtp_tokens": 1,
                "quantiles": quantiles,
                "use_cache": False,
            },
            "deterministic": True,
        }
        test_params.append(test_param_1)

        test_param_2 = test_param_1.copy()
        test_param_2.update(
            {
                "forward_kwargs": {"revin": True},
            }
        )
        test_params.append(test_param_2)

        return test_params
