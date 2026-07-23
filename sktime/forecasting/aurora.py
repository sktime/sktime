# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Aurora for forecasting."""

__all__ = ["AuroraForecaster"]

import contextlib
import io

import numpy as np
import pandas as pd

from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    FoundationModelSpec,
    ModelHandle,
)
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
rearrange = _safe_import("einops.rearrange")


class AuroraForecaster(BaseFoundationForecaster):
    """Zero-shot forecaster wrapping Aurora via the ``aurora-model`` package.

    Aurora is a multimodal time series foundation model supporting generative
    probabilistic forecasting. Besides multivariate support, it can optionally
    condition on free-text domain context and vision inputs.

    Inference follows the official ``aurora-model`` API [2]_ and Hugging Face
    model card [1]_ examples.

    Parameters
    ----------
    repo_id : str, default="DecisionIntelligence/Aurora"
        Hugging Face repository id for model weights.
    weights_filename : str, default="model.safetensors"
        Weights file name in the Hugging Face repository.
    cache_dir : str, optional, default=None
        Local cache directory for downloaded weights.
    force_download : bool, default=False
        Whether to force re-download of weights from the Hub.
    device : str, optional, default=None
        Device for inference. If ``None``, uses CUDA when available, else CPU.
    context_length : int, optional, default=None
        Number of trailing history steps passed to the model. If ``None``, uses
        the full series seen at predict time.
    inference_token_len : int, default=48
        Patch length for inference. Using the series period length when known
        is recommended.
    num_samples : int, default=100
        Number of stochastic forecast trajectories from flow matching. Point
        forecasts use the sample mean. Quantile forecasts require
        ``num_samples > 1``.
    max_text_length : int, default=200
        Maximum token length when ``text`` is provided.
    text : str, optional, default=None
        Optional text context for multimodal forecasting (e.g. domain metadata
        or event descriptions). The same text is applied to every target
        variable (channel-independent inference).
    vision : PIL.Image, array-like, or torch.Tensor, optional, default=None
        Optional external RGB image for multimodal conditioning. When
        ``None`` (default), Aurora renders a pseudo-image from the historical
        series internally (period-based 2D layout).

        Set ``vision`` only when a real image should provide additional
        context alongside the series (e.g. a chart or domain photograph).
        Accepted inputs are passed to Aurora's ViT preprocessor: a single
        ``PIL.Image``, a list of images (batch size must match the model
        batch: 1 for univariate, ``n_vars`` for multivariate), a numpy array,
        or a ``torch.Tensor`` of shape ``(batch, 3, H, W)`` in RGB channel
        order. Images are resized to 224x224 and normalized inside the
        model.

    References
    ----------
    .. [1] https://huggingface.co/DecisionIntelligence/Aurora
    .. [2] https://pypi.org/project/aurora-model/0.2.0/
    .. [3] https://arxiv.org/abs/2509.22295

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.aurora import AuroraForecaster
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, _ = temporal_train_test_split(y)
    >>> f = AuroraForecaster(num_samples=10)  # doctest: +SKIP
    >>> f.fit(y_train)  # doctest: +SKIP
    >>> y_pred = f.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Multimodal forecasting with text context:

    >>> f = AuroraForecaster(  # doctest: +SKIP
    ...     text="Monthly airline passenger totals.",
    ...     num_samples=10,
    ... )
    >>> f.fit(y_train)  # doctest: +SKIP
    >>> y_pred = f.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "authors": [
            "Faakhir30",
            # Aurora authors:
            "ccloud0525",
            "PengChen12",
            "Jianxin Jin",
            "Wanghui Qiu",
            "Yang Shu",
            "Bin Yang",
            "Chenjuan Guo",
        ],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.10",
        "python_dependencies": [
            "aurora-model==0.2.0",
            "torch",
            "einops",
            "transformers",
            "torchvision",
            "huggingface-hub",
            "safetensors",
        ],
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "requires-fh-in-fit": False,
        "property:randomness": "stochastic",
        "tests:vm": True,
        # since aurora is generative probabilistic
        # we skip the score test as it is not deterministic
        "tests:skip_by_name": ["test_score"],
    }

    def __init__(
        self,
        repo_id: str = "DecisionIntelligence/Aurora",
        weights_filename: str = "model.safetensors",
        cache_dir: str | None = None,
        force_download: bool = False,
        device: str | None = None,
        context_length: int | None = None,
        inference_token_len: int = 48,
        num_samples: int = 100,
        max_text_length: int = 200,
        text: str | None = None,
        vision=None,
    ):
        self.repo_id = repo_id
        self.weights_filename = weights_filename
        self.cache_dir = cache_dir
        self.force_download = force_download
        self.device = device
        self.context_length = context_length
        self.inference_token_len = inference_token_len
        self.num_samples = num_samples
        self.max_text_length = max_text_length
        self.text = text
        self.vision = vision

        model_spec = FoundationModelSpec(
            model_path=repo_id,
            device="auto" if device is None else device,
            load_extra_kwargs={
                "weights_filename": weights_filename,
                "cache_dir": cache_dir,
                "force_download": force_download,
            },
            predict_extra_kwargs={
                "context_length": context_length,
                "inference_token_len": inference_token_len,
                "num_samples": num_samples,
                "max_text_length": max_text_length,
                "text": text,
            },
        )
        super().__init__(model_spec=model_spec)

    def _load_model(self):
        """Load the Aurora checkpoint into a shared model handle."""
        from aurora import load_model

        model_spec = self.model_spec
        with contextlib.redirect_stdout(io.StringIO()):
            model = load_model(
                repo_id=model_spec.model_path,
                device=model_spec.device,
                **model_spec.load_extra_kwargs,
            )
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
        """Generate Aurora sample paths and normalize their summaries."""
        predict_kwargs = self.model_spec.predict_extra_kwargs
        context_length = predict_kwargs["context_length"]
        if context_length is not None and len(context_y) > context_length:
            context_y = context_y.iloc[-context_length:]
        if alpha is not None and predict_kwargs["num_samples"] < 2:
            raise ValueError(
                "Error in AuroraForecaster: Quantile prediction requires"
                "num_samples >= 2; got "
                f"num_samples={predict_kwargs['num_samples']}."
            )

        inputs, n_vars, _ = _prepare_context(
            y=context_y,
            device=self.model_spec.device,
        )
        text_kwargs = _prepare_text_kwargs(
            model=handle.model,
            text=predict_kwargs["text"],
            n_vars=n_vars,
            max_text_length=predict_kwargs["max_text_length"],
            device=self.model_spec.device,
        )
        vision_inputs = _prepare_vision_inputs(
            self.vision,
            self.model_spec.device,
        )

        generate_kwargs = {
            "inputs": inputs,
            "max_output_length": pred_len,
            "num_samples": predict_kwargs["num_samples"],
            "inference_token_len": predict_kwargs["inference_token_len"],
            **text_kwargs,
        }
        if vision_inputs is not None:
            generate_kwargs["vision_inputs"] = vision_inputs

        output = handle.model.generate(**generate_kwargs)
        samples = output.detach().cpu().numpy()
        mean = samples.mean(axis=1).T

        if alpha is None:
            return ForecastResult(mean=mean)

        quantiles = {
            float(quantile): np.quantile(samples, quantile, axis=1).T
            for quantile in alpha
        }
        return ForecastResult(mean=mean, quantiles=quantiles)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"num_samples": 5, "inference_token_len": 48},
            {"num_samples": 3, "text": "test context"},
        ]


def _as_dataframe(y):
    if isinstance(y, pd.Series):
        return y.to_frame()
    return y


def _prepare_context(y, device):
    """Build model inputs and metadata from ``y``.

    Returns
    -------
    inputs : torch.Tensor
        Shape ``(batch, lookback)`` for univariate or ``(n_vars, lookback)``
        for multivariate (channel-independent batching).
    n_vars : int
        Number of target variables.
    columns : list
        Column names from ``y``.
    """
    df = _as_dataframe(y)
    columns = list(df.columns)
    n_vars = len(columns)
    values = df.to_numpy(dtype=np.float32)

    if n_vars == 1:
        tensor = torch.tensor(values[:, 0], dtype=torch.float32, device=device)
        tensor = tensor.unsqueeze(0)
        return tensor, 1, columns

    # (time, vars) -> (1, time, vars) -> (vars, time)
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    tensor = tensor.unsqueeze(0)
    tensor = rearrange(tensor, "b l c -> (b c) l")
    return tensor, n_vars, columns


def _prepare_text_kwargs(model, text, n_vars, max_text_length, device):
    if text is None:
        return {}

    tokenizer = model.tokenizer
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    text_input_ids = tokenized["input_ids"].to(device)
    text_attention_mask = tokenized["attention_mask"].to(device)
    text_token_type_ids = tokenized.get(
        "token_type_ids", torch.zeros_like(text_input_ids)
    ).to(device)

    if n_vars > 1:
        text_input_ids = text_input_ids.repeat(n_vars, 1)
        text_attention_mask = text_attention_mask.repeat(n_vars, 1)
        text_token_type_ids = text_token_type_ids.repeat(n_vars, 1)

    return {
        "text_input_ids": text_input_ids,
        "text_attention_mask": text_attention_mask,
        "text_token_type_ids": text_token_type_ids,
    }


def _prepare_vision_inputs(vision, device):
    if vision is None:
        return None
    if torch.is_tensor(vision):
        return vision.to(device)
    # PIL images, lists of images, or numpy arrays — Aurora's ViTImageProcessor
    # handles resize/normalization inside VisionEncoder.process_real_image.
    return vision
