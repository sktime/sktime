# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interface for the YingLong foundation model for time series forecasting.

YingLong is a time series foundation model based on a decoder-only
Transformer architecture. This module provides an ``sktime`` forecasting
interface for the pretrained YingLong models available through Hugging Face.
"""

__author__ = ["Bot87Ever"]

__all__ = ["YingLongForecaster"]

import numpy as np
import pandas as pd

from skbase.utils.dependencies import _check_soft_dependencies
from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class YingLongForecaster(BaseForecaster):
    """YingLong zero-shot forecasting foundation model.

    YingLong is a pretrained time series foundation model that generates
    future observations autoregressively from a historical context window.

    Parameters
    ----------
    model_path : str, default="qcw2333/YingLong_6m"
        Hugging Face model repository identifier or local path.

        YingLong provides several pretrained model sizes. The default
        ``YingLong_6m`` checkpoint is the smallest released model.

    device : str, default="cuda"
        Device used for model inference, e.g. ``"cpu"``, ``"cuda"``,
        or ``"cuda:0"``.

    torch_dtype : str or torch.dtype, default="bfloat16"
        Data type used when loading the pretrained model.

        If a string is supplied, it must correspond to a valid attribute
        of ``torch``, for example ``"bfloat16"`` or ``"float32"``.

    trust_remote_code : bool, default=True
        Whether to allow Hugging Face Transformers to execute the custom
        model implementation supplied by the YingLong model repository.

        YingLong requires custom model code, so this is ``True`` by default.

    model_kwargs : dict or None, default=None
        Additional keyword arguments passed to
        ``AutoModelForCausalLM.from_pretrained``.

    ignore_deps : bool, default=False
        Whether to clear the estimator's soft dependency tags. This is
        primarily useful for testing without installing the optional
        YingLong dependencies.

    Notes
    -----
    YingLong expects the input context length to be compatible with the
    model's patch size. The wrapper validates this requirement using the
    configuration of the loaded checkpoint.

    YingLong is a zero-shot model and does not perform task-specific
    parameter fitting. ``fit`` stores the observed series and loads the
    pretrained model; ``predict`` performs autoregressive generation.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.yinglong import YingLongForecaster
    >>> y = load_airline()
    >>> forecaster = YingLongForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    References
    ----------
    .. [1] YingLong repository:
       https://github.com/wxie9/YingLong

    .. [2] YingLong model card:
       https://huggingface.co/qcw2333/YingLong_6m
    """

    _tags = {
        "authors": ["Bot87Ever"],
        "maintainers": ["Bot87Ever"],
        "python_dependencies": [
            "torch",
            "transformers",
            "xformers",
            "einops",
        ],
        "tests:vm": True,
        "capability:multivariate": False,
        "capability:exogenous": False,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pretrain": False,
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
    }

    def __init__(
        self,
        model_path="qcw2333/YingLong_6m",
        device="cuda",
        torch_dtype="bfloat16",
        trust_remote_code=True,
        model_kwargs=None,
        ignore_deps=False,
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs
        self.ignore_deps = ignore_deps

        super().__init__()

    def __dynamic_tags__(self):
        """Set tags that depend on constructor parameters."""
        if self.ignore_deps:
            self.set_tags(python_dependencies=[])

    def _get_unique_key(self):
        """Return the cache key for the pretrained YingLong model."""
        return str(
            sorted(
                {
                    "model_path": self.model_path,
                    "device": self.device,
                    "torch_dtype": str(self.torch_dtype),
                    "trust_remote_code": self.trust_remote_code,
                    "model_kwargs": (
                        None
                        if self.model_kwargs is None
                        else tuple(sorted(self.model_kwargs.items()))
                    ),
                }.items()
            )
        )

    def _load_model(self):
        """Load the pretrained YingLong model."""
        _check_soft_dependencies(
            "torch",
            severity="error",
            obj=self,
        )
        _check_soft_dependencies(
            "transformers",
            severity="error",
            obj=self,
        )
        _check_soft_dependencies(
            "xformers",
            severity="error",
            obj=self,
        )
        _check_soft_dependencies(
            "einops",
            severity="error",
            obj=self,
        )

        cached = _CachedYingLong(
            key=self._get_unique_key(),
            model_path=self.model_path,
            device=self.device,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            model_kwargs=self.model_kwargs,
        )

        return cached.load()

    def __getstate__(self):
        """Return state suitable for pickling."""
        state = self.__dict__.copy()

        if "model_" in state:
            state["model_"] = None

        return state

    def __setstate__(self, state):
        """Restore estimator state after unpickling."""
        self.__dict__.update(state)

    def _fit(self, y, X=None, fh=None):
        """Fit the YingLong forecaster.

        YingLong is a zero-shot foundation model. Fitting therefore stores
        the observed context and loads the pretrained model without updating
        model parameters.
        """
        self.context_ = y.copy()
        self.y_name_ = y.name

        self.model_ = self._load_model()

        self.patch_size_ = int(
            getattr(self.model_.config, "patch_size", 32)
        )
        self.block_size_ = int(
            getattr(self.model_.config, "block_size", 8224)
        )

        if self.patch_size_ <= 0:
            raise ValueError(
                "YingLong model configuration has an invalid patch size."
            )

        if self.block_size_ <= 0:
            raise ValueError(
                "YingLong model configuration has an invalid block size."
            )

        if len(self.context_) < self.patch_size_:
            raise ValueError(
                "YingLong requires at least "
                f"{self.patch_size_} observations in the input context; "
                f"found {len(self.context_)}."
            )

        if len(self.context_) % self.patch_size_ != 0:
            raise ValueError(
                "The YingLong input context length must be a multiple of "
                f"the model patch size ({self.patch_size_}); "
                f"found {len(self.context_)} observations."
            )

        return self

    def _predict(self, fh, X=None):
        """Generate point forecasts using YingLong."""
        _check_soft_dependencies(
            "torch",
            severity="error",
            obj=self,
        )

        import torch

        fh_relative = fh.to_relative(self.cutoff)
        fh_values = np.asarray(fh_relative._values, dtype=int)

        if np.any(fh_values <= 0):
            raise ValueError(
                "YingLongForecaster only supports out-of-sample "
                "forecasting horizons."
            )

        prediction_length = int(np.max(fh_values))

        context = self.context_.to_numpy(dtype=np.float32)

        required_length = len(context) + (
            (prediction_length // self.patch_size_) + 1
        ) * self.patch_size_

        if required_length > self.block_size_:
            raise ValueError(
                "The requested forecast exceeds the YingLong model's "
                f"maximum sequence length. The model block size is "
                f"{self.block_size_}, but the requested context and "
                f"forecast require {required_length} positions."
            )

        input_tensor = torch.from_numpy(context).unsqueeze(0)

        if isinstance(self.torch_dtype, str):
            try:
                dtype = getattr(torch, self.torch_dtype)
            except AttributeError as exc:
                raise ValueError(
                    f"Unknown torch dtype: {self.torch_dtype!r}"
                ) from exc
        else:
            dtype = self.torch_dtype

        input_tensor = input_tensor.to(
            device=self.device,
            dtype=dtype,
        )

        if self.model_ is None:
            self.model_ = self._load_model()

        with torch.no_grad():
            output = self.model_.generate(
                input_tensor,
                future_token=prediction_length,
            )

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "YingLong returned an unexpected output type. "
                "Expected a torch.Tensor."
            )

        if output.ndim == 3:
            if output.shape[0] != 1 or output.shape[2] != 1:
                raise ValueError(
                    "YingLong returned an unexpected output shape: "
                    f"{tuple(output.shape)}."
                )
            predictions = output[0, :, 0]
        elif output.ndim == 2:
            if output.shape[0] != 1:
                raise ValueError(
                    "YingLong returned an unexpected output shape: "
                    f"{tuple(output.shape)}."
                )
            predictions = output[0]
        else:
            raise ValueError(
                "YingLong returned an unexpected output shape: "
                f"{tuple(output.shape)}."
            )

        if len(predictions) < prediction_length:
            raise ValueError(
                "YingLong returned fewer predictions than requested: "
                f"{len(predictions)} instead of {prediction_length}."
            )

        predictions = (
            predictions.detach()
            .float()
            .cpu()
            .numpy()
        )

        predictions = predictions[fh_values - 1]

        index = fh.to_absolute_index(self.cutoff)

        return pd.Series(
            predictions,
            index=index,
            name=self.y_name_,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {
                "model_path": "qcw2333/YingLong_6m",
                "device": "cuda",
                "torch_dtype": "bfloat16",
                "ignore_deps": True,
            }
        ]


@_multiton
class _CachedYingLong:
    """Cache wrapper for pretrained YingLong models."""

    def __init__(
        self,
        key,
        model_path,
        device,
        torch_dtype,
        trust_remote_code,
        model_kwargs,
    ):
        self.key = key
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs
        self.model_ = None

    def load(self):
        """Load or return the cached YingLong model."""
        if self.model_ is not None:
            return self.model_

        _check_soft_dependencies(
            "torch",
            severity="error",
        )
        _check_soft_dependencies(
            "transformers",
            severity="error",
        )

        import torch
        from transformers import AutoModelForCausalLM

        if isinstance(self.torch_dtype, str):
            try:
                torch_dtype = getattr(torch, self.torch_dtype)
            except AttributeError as exc:
                raise ValueError(
                    f"Unknown torch dtype: {self.torch_dtype!r}"
                ) from exc
        else:
            torch_dtype = self.torch_dtype

        model_kwargs = (
            {}
            if self.model_kwargs is None
            else dict(self.model_kwargs)
        )

        model_kwargs.setdefault(
            "trust_remote_code",
            self.trust_remote_code,
        )

        if torch_dtype is not None:
            model_kwargs.setdefault(
                "torch_dtype",
                torch_dtype,
            )

        self.model_ = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )

        self.model_.to(self.device)
        self.model_.eval()

        return self.model_