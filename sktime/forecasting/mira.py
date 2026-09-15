# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements MIRA for forecasting."""

__all__ = ["MIRAForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    FoundationModelSpec,
    ModelHandle,
)
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")


class MIRAForecaster(BaseFoundationForecaster):
    """Zero-shot forecaster wrapping Microsoft MIRA via vendored ``sktime.libs.mira``.

    MIRA is a foundation model for medical time series, supporting zero-shot
    forecasting on irregularly sampled clinical signals via continuous-time
    rotary positional encoding and neural ODE extrapolation.

    Univariate only. Inference follows the autoregressive loop in the official
    MIRA repository [2]_.

    Parameters
    ----------
    model_path : str, default="MIRA-Mode/MIRA"
        Hugging Face model id or local path.
    revision : str, default="main"
        Model revision on the Hugging Face Hub.
    config : dict, optional, default=None
        Extra kwargs for ``MIRAForPrediction.from_pretrained``.
    context_length : int, optional, default=None
        Number of history steps passed to the model. If ``None``, uses the full
        series seen at predict time.
    time_alpha : float, default=1.0
        ``alpha`` used for CT-RoPE normalization of time values.
    time_snap_step : float, default=0.1
        ``snap_step`` used for CT-RoPE normalization of time values.

    References
    ----------
    .. [1] https://huggingface.co/MIRA-Mode/MIRA
    .. [2] https://github.com/microsoft/MIRA
    .. [3] https://arxiv.org/abs/2506.07584

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.mira import MIRAForecaster
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, _ = temporal_train_test_split(y)
    >>> f = MIRAForecaster()  # doctest: +SKIP
    >>> f.fit(y_train)  # doctest: +SKIP
    >>> y_pred = f.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "authors": [
            "Faakhir30",
            # from Microsoft:
            "HaoBytes",
            "chang-xu-ms",
            "Bowen Deng",
            "Zhiyuan Feng",
            "Viktor Schlegel",
            "Yu-Hao Huang",
            "Yizheng Sun",
            "Jingyuan Sun",
            "Kailai Yang",
            "Yiyao Yu",
            "Jiang Bian",
        ],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.10",
        "python_dependencies": [
            "torch",
            "torchdiffeq",
            "transformers==4.40.1",
            "accelerate==0.28.0",
        ],
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:unequal_length": True,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "requires-fh-in-fit": False,
        "tests:vm": True,
        "tests:libs": ["sktime.libs.mira"],
    }

    def __init__(
        self,
        model_path: str = "MIRA-Mode/MIRA",
        revision: str = "main",
        config: dict | None = None,
        context_length: int | None = None,
        time_alpha: float = 1.0,
        time_snap_step: float = 0.1,
    ):
        self.model_path = model_path
        self.revision = revision
        self.config = config
        self.context_length = context_length
        self.time_alpha = time_alpha
        self.time_snap_step = time_snap_step

        model_spec = FoundationModelSpec(
            model_path=model_path,
            revision=revision,
            config=config,
            device="auto",
            predict_extra_kwargs={
                "context_length": context_length,
                "time_alpha": time_alpha,
                "time_snap_step": time_snap_step,
            },
        )
        super().__init__(model_spec=model_spec)

    def _load_model(self):
        """Load MIRA from the vendored implementation into a shared handle."""
        from sktime.libs.mira import MIRAForPrediction

        model_spec = self.model_spec
        load_kwargs = {"revision": model_spec.revision}
        load_kwargs.update(model_spec.config)
        model = MIRAForPrediction.from_pretrained(
            model_spec.model_path,
            **load_kwargs,
        )
        return ModelHandle(model=model.to(model_spec.device))

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
        """Run MIRA's normalized autoregressive rollout."""
        predict_kwargs = self.model_spec.predict_extra_kwargs
        values, times = _prepare_context(
            y=context_y,
            pred_len=pred_len,
            cutoff=self.cutoff,
            context_length=predict_kwargs["context_length"],
        )
        values = values.to(self.model_spec.device)
        times = times.to(self.model_spec.device)
        context_len = values.shape[1]

        from sktime.libs.mira.mira_inference import mira_predict_autoregressive_norm
        from sktime.libs.mira.utils_time_normalization import normalize_time_for_ctrope

        # CT-RoPE scaling
        times, _, _ = normalize_time_for_ctrope(
            time_values=times,
            attention_mask=torch.ones_like(times),
            seq_length=times.shape[1],
            alpha=predict_kwargs["time_alpha"],
            snap_step=predict_kwargs["time_snap_step"],
        )

        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True) + 1e-6

        preds = mira_predict_autoregressive_norm(
            handle.model,
            values,
            times,
            context_len,
            pred_len,
            mean,
            std,
        )

        values_out = preds.detach().cpu().numpy()
        if values_out.ndim == 1:
            values_out = values_out[:, np.newaxis]
        return ForecastResult(mean=values_out)

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
        return [{}, {"time_alpha": 0.5}]


def _index_as_float(index) -> np.ndarray:
    """Map sktime index labels to float time coordinates for MIRA ``time_values``."""
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return index.asi8.astype(np.float64)
    arr = np.asarray(index)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float64)
    return np.arange(len(index), dtype=np.float64)


def _prepare_context(y, pred_len, cutoff, context_length):
    """Build ``values`` and ``times`` tensors expected by ``mira_inference``.

    Parameters
    ----------
    y : pd.DataFrame
        Context series passed. Must be univariate;
    pred_len : int
        Autoregressive rollout length; future timestamps are built for relative
        steps: 1 .. pred_len.
    cutoff : pandas index element or compatible
        Last time point of the context series.
    context_length : int or None
        Number of trailing history steps to keep. If ``None``, all of ``y`` is used.

    Returns
    -------
    values : torch.Tensor
        Float32 tensor of shape (1, context_len)
    times : torch.Tensor
        Float32 tensor of shape (1, context_len + pred_len).
    """
    if isinstance(y, pd.Series):
        y = y.to_frame()
    series = y.iloc[:, 0].to_numpy(dtype=np.float32)
    index = y.index
    future_index = (
        ForecastingHorizon(range(1, pred_len + 1)).to_absolute(cutoff)._values
    )

    if context_length is not None and len(series) > context_length:
        series = series[-context_length:]
        index = index[-context_length:]

    times_np = np.concatenate([_index_as_float(index), _index_as_float(future_index)])
    values = torch.tensor(series.reshape(1, -1), dtype=torch.float32)
    times = torch.tensor(times_np.reshape(1, -1), dtype=torch.float32)
    return values, times
