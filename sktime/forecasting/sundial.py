# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Sundial forecaster for ``sktime``."""

__author__ = ["WenWeiTHU", "geetu040"]
# WenWeiTHU for thuml/sundial-base-128m

__all__ = ["SundialForecaster"]

from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class SundialForecaster(BaseForecaster):
    """Sundial zero-shot forecaster via Hugging Face ``transformers``.

    This forecaster wraps Sundial [1]_, [2]_, [3]_ and exposes zero-shot
    forecasting through the ``sktime`` forecasting interface. Calling
    :meth:`fit` loads the model and stores the observed series as forecasting
    context; it does not train or fine-tune model weights.

    Sundial generates one or more sample paths. Point forecasts are computed as
    the empirical mean over generated samples. Quantile forecasts are computed
    as empirical quantiles over generated samples.

    Parameters
    ----------
    model_path : str, default="thuml/sundial-base-128m"
        Hugging Face repository identifier or local path to a Sundial
        checkpoint. If ``None``, a model is created from ``config`` with random
        weights, mainly for tests and local experimentation.
    config : SundialConfig or dict, optional (default=None)
        Model configuration used when ``model_path=None``. If provided as a
        ``dict``, it is converted with ``SundialConfig.from_dict``. If ``None``
        and ``model_path=None``, the default ``SundialConfig`` is used.
    device_map : str, dict, int, or torch.device, default="cpu"
        Device placement following the ``transformers`` ``device_map`` naming
        convention, for example ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or
        ``"auto"``.
    dtype : torch.dtype or str, optional (default=None)
        Data type used for model loading, following the ``transformers``
        ``dtype`` convention, for example ``torch.float16``,
        ``torch.bfloat16``, or ``"auto"``.
    forward_kwargs : dict, optional (default=None)
        Additional keyword arguments forwarded to ``model.generate(...)`` during
        prediction, for example ``{"num_samples": 20, "revin": True}``.
    deterministic : bool, default=False
        Whether predictions should reset the ``transformers`` random seed before
        generation.

    References
    ----------
    .. [1] Sundial: A Family of Highly Capable Time Series Foundation Models:
       https://arxiv.org/abs/2502.00816
    .. [2] Sundial repository:
       https://github.com/thuml/Sundial
    .. [3] Sundial model card:
       https://huggingface.co/thuml/sundial-base-128m

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster
    >>> y = load_airline()
    >>> forecaster = SundialForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Running with explicit device, dtype, and sampling settings:

    >>> import torch
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster
    >>> y = load_airline()
    >>> forecaster = SundialForecaster(  # doctest: +SKIP
    ...     device_map="cuda",
    ...     dtype=torch.bfloat16,
    ...     forward_kwargs={"num_samples": 20},
    ...     deterministic=True,
    ... )
    >>> y_pred = forecaster.fit(y).predict(fh=[1, 2, 3])  # doctest: +SKIP

    Passing generation options through ``forward_kwargs``:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster
    >>> y = load_airline()
    >>> forecaster = SundialForecaster(  # doctest: +SKIP
    ...     forward_kwargs={"num_samples": 20, "revin": False},
    ... )
    >>> y_pred = forecaster.fit(y).predict(fh=[1, 2, 3])  # doctest: +SKIP

    Quantile prediction from generated samples:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster
    >>> y = load_airline()
    >>> forecaster = SundialForecaster(  # doctest: +SKIP
    ...     forward_kwargs={"num_samples": 50},
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict_quantiles(  # doctest: +SKIP
    ...     fh=[1, 2, 3],
    ...     alpha=[0.1, 0.5, 0.9],
    ... )
    """

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "authors": ["WenWeiTHU", "geetu040"],
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers[torch]>=4.40.1,<5.0.0"],
        "tests:vm": True,
        "tests:libs": ["sktime.libs.sundial"],
    }

    def __init__(
        self,
        model_path="thuml/sundial-base-128m",
        config=None,
        device_map="cpu",
        dtype=None,
        forward_kwargs=None,
        deterministic=False,
    ):
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.dtype = dtype
        self.forward_kwargs = forward_kwargs
        self.deterministic = deterministic

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Load the model and store history for zero-shot forecasting."""
        self.model_ = self._load_model()
        self.context_ = y

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon."""
        samples, fh, preds_idx = self._generate_samples(fh)

        preds = samples.mean(axis=1)
        preds = preds[:, preds_idx].T
        preds = pd.DataFrame(
            preds,
            index=fh.to_absolute(self._cutoff)._values,
            columns=self.context_.columns,
        )

        return preds

    def _predict_quantiles(self, fh, X, alpha):
        """Compute empirical prediction quantiles from generated samples."""
        samples, fh, preds_idx = self._generate_samples(fh)

        if alpha is None:
            alpha = [0.1, 0.5, 0.9]
        alpha = [round(i, 3) for i in alpha]

        preds = np.quantile(samples, q=alpha, axis=1)
        preds = np.moveaxis(preds, 0, -1)
        preds = preds[:, preds_idx, :]
        preds = preds.transpose(1, 0, 2).reshape(len(preds_idx), -1)

        columns = pd.MultiIndex.from_product([self.context_.columns, alpha])
        pred_quantiles = pd.DataFrame(
            data=preds,
            index=fh.to_absolute(self._cutoff)._values,
            columns=columns,
        )

        return pred_quantiles

    def _generate_samples(self, fh):
        """Generate Sundial sample paths for the requested horizon."""
        import torch
        import transformers

        if self.deterministic:
            transformers.set_seed(42)

        self.model_ = self._load_model()

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1
        horizon_length = np.max(preds_idx) + 1
        output_token_lens = getattr(self.model_.config, "output_token_lens", [])
        if output_token_lens and max(output_token_lens) < horizon_length:
            raise ValueError(
                "Requested forecasting horizon exceeds Sundial model capacity: "
                f"max requested step={horizon_length}, "
                f"max output_token_lens={max(output_token_lens)}. "
                "Use a shorter horizon or a model/config with a larger "
                "output_token_lens value."
            )

        past_values = self.context_.to_numpy().T
        past_values = torch.from_numpy(past_values)
        past_values = past_values.to(self.model_.dtype)
        past_values = past_values.to(self.model_.device)

        forward_kwargs = {} if not self.forward_kwargs else self.forward_kwargs
        output = self.model_.generate(
            past_values,
            max_new_tokens=horizon_length,
            **forward_kwargs,
        )

        samples = output.detach().float().cpu().numpy()

        return samples, fh, preds_idx

    def _load_model(self):
        """Load or retrieve a cached Sundial model instance."""
        if hasattr(self, "model_") and self.model_ is not None:
            return self.model_

        self.model_ = _CachedSundial(
            key=self._get_unique_key(),
            model_path=self.model_path,
            config=self.config,
            device_map=self.device_map,
            dtype=self.dtype,
        ).load()

        return self.model_

    def _get_unique_key(self):
        """Build cache key for the multiton model loader."""
        key = {
            "model_path": self.model_path,
            "config": self.config,
            "device_map": self.device_map,
            "dtype": self.dtype,
        }
        return str(sorted(key.items()))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        config = {
            "input_token_len": 2,
            "hidden_size": 4,
            "intermediate_size": 8,
            "output_token_lens": [8],
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "max_position_embeddings": 64,
            "flow_loss_depth": 1,
            "num_sampling_steps": 1,
            "diffusion_batch_mul": 1,
            "use_cache": False,
        }
        return [
            {
                "model_path": None,
                "config": config,
                "device_map": "cpu",
                "forward_kwargs": {"num_samples": 2},
                "deterministic": True,
            },
            {
                "model_path": None,
                "config": config,
                "device_map": "cpu",
                "forward_kwargs": {"num_samples": 3, "revin": False},
                "deterministic": True,
            },
            {
                "model_path": None,
                "config": config,
                "device_map": None,
                "forward_kwargs": {"num_samples": 1, "revin": True},
                "deterministic": True,
            },
        ]


@_multiton
class _CachedSundial:
    """Multiton-backed cache wrapper for a loaded Sundial model."""

    def __init__(
        self,
        key,
        model_path,
        config,
        device_map,
        dtype,
    ):
        self.key = key
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.dtype = dtype
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
        """Load Sundial model weights from ``self.model_path``."""
        from sktime.libs.sundial import SundialConfig, SundialForPrediction

        kwargs = {}
        if self.config is not None:
            config = deepcopy(self.config)
            if isinstance(config, dict):
                config = SundialConfig.from_dict(config)
            kwargs["config"] = config
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        if self.dtype is not None:
            kwargs["torch_dtype"] = self.dtype

        model = SundialForPrediction.from_pretrained(self.model_path, **kwargs)

        return model

    def _load_randomly(self):
        """Initialize a Sundial model randomly from config."""
        from sktime.libs.sundial import SundialConfig, SundialForPrediction

        warn(
            "Initializing Sundial from config creates random weights. Sundial "
            "training is not supported by this estimator, so these weights "
            "will stay random and are only suitable for tests or local "
            "experimentation.",
            UserWarning,
            stacklevel=2,
        )

        config = deepcopy(self.config)
        if not config:
            config = SundialConfig()
        if isinstance(config, dict):
            config = SundialConfig.from_dict(config)

        model = SundialForPrediction(config)
        if self.dtype is not None:
            model = model.to(dtype=self.dtype)
        if self.device_map is not None:
            model = model.to(self.device_map)

        return model
