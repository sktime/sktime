# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements MIRA for forecasting."""

__all__ = ["MIRAForecaster"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.dependencies import _safe_import
from sktime.utils.singleton import _multiton

torch = _safe_import("torch")


class MIRAForecaster(BaseForecaster):
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
        self._config = {} if self.config is None else self.config.copy()
        self._device = _resolve_device()

    def _get_unique_model_key(self):
        key_items = {
            "model_path": self.model_path,
            "revision": self.revision,
            "device": self._device,
            **self._config,
        }
        return str(sorted(key_items.items()))

    def _load_model(self):
        return _CachedMIRA(
            key=self._get_unique_model_key(),
            forecaster=self,
        ).load_from_checkpoint()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Loads the pretrained MIRA checkpoint and stores ``y`` as context
        for zero-shot prediction.

        Parameters
        ----------
        y : pd.DataFrame
            Endogenous time series (univariate, one column).
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

    def _predict(self, fh, X=None):
        if self.model is None:
            self.model = self._load_model()
            self.model.eval()

        if fh is None:
            fh = self.fh
        fh_rel = fh.to_relative(self.cutoff)
        pred_len = int(np.max(fh_rel.to_numpy()))

        values, times = _prepare_context(
            y=self._y,
            pred_len=pred_len,
            cutoff=self.cutoff,
            context_length=self.context_length,
        )
        context_len = values.shape[1]

        from sktime.libs.mira.mira_inference import mira_predict_autoregressive_norm
        from sktime.libs.mira.utils_time_normalization import normalize_time_for_ctrope

        # CT-RoPE scaling
        times, _, _ = normalize_time_for_ctrope(
            time_values=times,
            attention_mask=torch.ones_like(times),
            seq_length=times.shape[1],
            alpha=self.time_alpha,
            snap_step=self.time_snap_step,
        )

        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True) + 1e-6

        preds = mira_predict_autoregressive_norm(
            self.model,
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

        pred_len_out = values_out.shape[0]
        index = (
            ForecastingHorizon(range(1, pred_len_out + 1))
            .to_absolute(self._cutoff)
            ._values
        )
        pred_df = pd.DataFrame(values_out, index=index, columns=self._y.columns)
        pred_df.index.names = self._y.index.names
        pred_out = fh_rel.get_expected_pred_idx(self._y, cutoff=self.cutoff)
        return pred_df.loc[pred_df.index.isin(pred_out)]

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


@_multiton
class _CachedMIRA:
    """Cached MIRA model; shared across forecaster instances with the same key."""

    def __init__(self, key: str, forecaster: "MIRAForecaster"):
        self.key = key
        self.forecaster = forecaster
        self.model = None

    def load_from_checkpoint(self):
        if self.model is not None:
            return self.model

        from sktime.libs.mira import MIRAForPrediction

        f = self.forecaster
        kwargs = {"revision": f.revision}
        kwargs.update(f._config)
        self.model = MIRAForPrediction.from_pretrained(f.model_path, **kwargs)
        return self.model.to(f._device)


def _resolve_device():
    if _check_soft_dependencies("torch", severity="none"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


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
