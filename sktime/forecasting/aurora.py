# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Aurora for forecasting."""

__all__ = ["AuroraForecaster"]

import contextlib
import io

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _safe_import
from sktime.utils.singleton import _multiton

torch = _safe_import("torch")
rearrange = _safe_import("einops.rearrange")


class AuroraForecaster(BaseForecaster):
    """Zero-shot forecaster wrapping Aurora via the ``aurora-model`` package.

    Aurora is a multimodal time series foundation model supporting generative
    probabilistic forecasting. Besides univariate and multivariate series, it
    can optionally condition on free-text domain context and (advanced) vision
    inputs. Text is not passed as exogenous time series ``X``; use the
    dedicated ``text`` parameter instead.

    Inference follows the official ``aurora-model`` API [2]_ and Hugging Face
    model card [1]_.

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
        Patch length for inference. The authors recommend using the series
        period length when known.
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
    vision : array-like, optional, default=None
        Optional real-image vision input forwarded as ``vision_inputs`` to
        ``model.generate``. When ``None``, Aurora derives pseudo-images from
        the time series internally.
    generate_kwargs : dict, optional, default=None
        Extra keyword arguments forwarded to ``model.generate``.

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
            "Xingjian Wu",
            "Jianxin Jin",
            "Wanghui Qiu",
            "Peng Chen",
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
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
        "requires-fh-in-fit": False,
        "property:randomness": "stochastic",
        "tests:vm": True,
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
        generate_kwargs: dict | None = None,
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
        self.generate_kwargs = generate_kwargs
        self.model = None
        super().__init__()

    def __getstate__(self):
        """Get state for pickling."""
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        """Set state for unpickling."""
        self.__dict__.update(state)

    def __post_init__(self):
        """Post-initialization setup."""
        self._generate_kwargs = (
            {} if self.generate_kwargs is None else self.generate_kwargs.copy()
        )
        self._device = _resolve_device(self.device)

    def _get_unique_model_key(self):
        key_items = {
            "repo_id": self.repo_id,
            "weights_filename": self.weights_filename,
            "cache_dir": self.cache_dir,
            "force_download": self.force_download,
            "device": self._device,
        }
        return str(sorted(key_items.items()))

    def _load_model(self):
        return _CachedAurora(
            key=self._get_unique_model_key(),
            forecaster=self,
        ).load_from_checkpoint()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Loads the pretrained Aurora checkpoint and stores ``y`` as context
        for zero-shot prediction.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Endogenous time series.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables. Ignored.
        fh : ForecastingHorizon, optional (default=None)
            Forecasting horizon.

        Returns
        -------
        self
        """
        self.model = self._load_model()
        self.model.eval()
        return self

    def predict(self, fh=None, X=None, y=None):
        """Forecast time series at future horizon.

        If ``y`` is not None, global forecast will be performed via
        ``fit_predict`` using the extended history in ``y``.
        """
        if y is not None:
            _fh = fh if self._fh is None and fh is not None else self._fh
            return self.fit_predict(fh=_fh, X=X, y=y)
        return super().predict(fh=fh, X=X)

    def _predict(self, fh, X=None):
        if self.model is None:
            self.model = self._load_model()
            self.model.eval()

        if fh is None:
            fh = self.fh
        fh_rel = fh.to_relative(self.cutoff)
        pred_len = int(np.max(fh_rel.to_numpy()))

        inputs, n_vars, columns = _prepare_context(
            y=self._y,
            context_length=self.context_length,
            device=self._device,
        )
        text_kwargs = _prepare_text_kwargs(
            model=self.model,
            text=self.text,
            n_vars=n_vars,
            max_text_length=self.max_text_length,
            device=self._device,
        )
        vision_inputs = _prepare_vision_inputs(self.vision, self._device)

        generate_kwargs = {
            "inputs": inputs,
            "max_output_length": pred_len,
            "num_samples": self.num_samples,
            "inference_token_len": self.inference_token_len,
            **text_kwargs,
            **self._generate_kwargs,
        }
        if vision_inputs is not None:
            generate_kwargs["vision_inputs"] = vision_inputs

        with torch.inference_mode():
            output = self.model.generate(**generate_kwargs)

        point = output.mean(dim=1).detach().cpu().numpy()
        rel_idx = (fh_rel.to_numpy() - 1).astype(int)
        values = point[:, rel_idx]

        index = fh.to_absolute(self._cutoff)._values
        pred_df = pd.DataFrame(values, index=index, columns=columns)
        pred_df.index.names = self._y.index.names
        return pred_df

    def _predict_quantiles(self, fh, X, alpha):
        if self.num_samples < 2:
            raise ValueError(
                "Quantile prediction requires num_samples >= 2; "
                f"got num_samples={self.num_samples}."
            )

        if self.model is None:
            self.model = self._load_model()
            self.model.eval()

        if fh is None:
            fh = self.fh
        fh_rel = fh.to_relative(self.cutoff)
        pred_len = int(np.max(fh_rel.to_numpy()))

        inputs, n_vars, columns = _prepare_context(
            y=self._y,
            context_length=self.context_length,
            device=self._device,
        )
        text_kwargs = _prepare_text_kwargs(
            model=self.model,
            text=self.text,
            n_vars=n_vars,
            max_text_length=self.max_text_length,
            device=self._device,
        )
        vision_inputs = _prepare_vision_inputs(self.vision, self._device)

        generate_kwargs = {
            "inputs": inputs,
            "max_output_length": pred_len,
            "num_samples": self.num_samples,
            "inference_token_len": self.inference_token_len,
            **text_kwargs,
            **self._generate_kwargs,
        }
        if vision_inputs is not None:
            generate_kwargs["vision_inputs"] = vision_inputs

        with torch.inference_mode():
            output = self.model.generate(**generate_kwargs)

        samples = output.detach().cpu().numpy()
        rel_idx = (fh_rel.to_numpy() - 1).astype(int)
        alpha = [float(a) for a in alpha]

        if n_vars == 1:
            qvals = np.quantile(samples[0, :, rel_idx], alpha, axis=0).T
            index = fh.to_absolute(self._cutoff)._values
            name = columns[0]
            return pd.DataFrame(
                qvals,
                index=index,
                columns=pd.MultiIndex.from_product([[name], alpha]),
            )

        frames = []
        index = fh.to_absolute(self._cutoff)._values
        for var_idx, col in enumerate(columns):
            qvals = np.quantile(samples[var_idx, :, rel_idx], alpha, axis=0).T
            frames.append(
                pd.DataFrame(
                    qvals,
                    index=index,
                    columns=pd.MultiIndex.from_product([[col], alpha]),
                )
            )
        return pd.concat(frames, axis=1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"num_samples": 5, "inference_token_len": 48},
            {"num_samples": 3, "text": "test context"},
        ]


@_multiton
class _CachedAurora:
    """Cached Aurora model; shared across forecaster instances with the same key."""

    def __init__(self, key: str, forecaster: "AuroraForecaster"):
        self.key = key
        self.forecaster = forecaster
        self.model = None

    def load_from_checkpoint(self):
        if self.model is not None:
            return self.model

        from aurora import load_model

        f = self.forecaster
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = load_model(
                repo_id=f.repo_id,
                weights_filename=f.weights_filename,
                cache_dir=f.cache_dir,
                force_download=f.force_download,
                device=f._device,
            )
        return self.model


def _resolve_device(device):
    if device is not None:
        return device
    if _check_soft_dependencies("torch", severity="none"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _as_dataframe(y):
    if isinstance(y, pd.Series):
        return y.to_frame()
    return y


def _prepare_context(y, context_length, device):
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
    if context_length is not None and len(df) > context_length:
        df = df.iloc[-context_length:]

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
    return torch.as_tensor(vision, dtype=torch.float32, device=device)
