"""Implements Time-LLM forecaster."""

__all__ = ["TimeLLMForecaster"]
__author__ = ["KimMeen", "jgyasu"]
# KimMeen for [ICLR 2024] Official implementation of Time-LLM

from types import SimpleNamespace
from typing import Optional

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _safe_import
from sktime.utils.singleton import _multiton

torch = _safe_import("torch")


class TimeLLMForecaster(BaseForecaster):
    """
    Interface to the Time-LLM.

    Time-LLM is a reprogramming framework
    to repurpose LLMs for general time series forecasting
    with the backbone language models kept intact. This method has been
    proposed in [2]_ and official code is given at [1]_.

    Parameters
    ----------
    task_name : str, default='long_term_forecast'
        Task to perform - can be one of ['long_term_forecast', 'short_term_forecast'].
    pred_len : int, default=24
        Forecast horizon - number of time steps to predict.
    seq_len : int, default=96
        Length of input sequence.
    llm_model : str, default='GPT2'
        LLM model to use - can be one of ['GPT2', 'LLAMA', 'BERT'].
    llm_layers : int, default=3
        Number of transformer layers to use from LLM.
    patch_len : int, default=16
        Length of patches for patch embedding.
    stride : int, default=8
        Stride between patches.
    d_model : int, default=128
        Model dimension.
    d_ff : int, default=128
        Feed-forward dimension.
    n_heads : int, default=4
        Number of attention heads.
    dropout : float, default=0.1
        Dropout rate.
    device : str, default='cuda' if available else 'cpu'
        Device to run model on.

    References
    ----------
    .. [1] https://github.com/KimMeen/Time-LLM
    .. [2] Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang,
    Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, Qingsong Wen.
    Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.
    https://arxiv.org/abs/2310.01728.

    Examples
    --------
    >>> from sktime.forecasting.time_llm import TimeLLMForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = TimeLLMForecaster(
    ...     pred_len=36,
    ...     seq_len=96,
    ...     llm_model='GPT2'
    ... )
    >>> forecaster.fit(y, fh=[1])
    TimeLLMForecaster(pred_len=1)
    >>> y_pred = forecaster.predict(fh=[1])
    """

    _tags = {
        "scitype:y": "univariate",
        "authors": ["KimMeen", "jgyasu"],
        # KimMeen for [ICLR 2024] Official implementation of Time-LLM
        "maintainers": ["jgyasu"],
        "python_dependencies": ["torch", "transformers"],
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
    }

    def __init__(
        self,
        task_name="long_term_forecast",
        pred_len=24,
        seq_len=96,
        llm_model="GPT2",
        llm_layers=3,
        llm_dim=768,
        patch_len=16,
        stride=8,
        d_model=128,
        d_ff=128,
        n_heads=4,
        dropout=0.1,
        device: Optional[str] = None,
        prompt_domain=False,
    ):
        self.task_name = task_name
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.llm_model = llm_model
        self.llm_layers = llm_layers
        self.llm_dim = llm_dim
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = dropout
        self.device = device
        self.prompt_domain = prompt_domain

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit
        """
        self.device_ = (
            "cuda" if self.device is None and torch.cuda.is_available() else "cpu"
        )

        self.fh_ = fh

        if isinstance(fh, int):
            self._pred_len = fh
        elif hasattr(fh, "__len__"):
            self._pred_len = len(fh)
        else:
            self._pred_len = self.pred_len

        # Create a unique key for the current model configuration
        key = self._get_unique_time_llm_key()

        # Load or reuse cached model with the same key
        self.model_ = _CachedTimeLLM(
            key=key,
            time_llm_kwargs={
                "task_name": self.task_name,
                "pred_len": self._pred_len,
                "seq_len": self.seq_len,
                "llm_model": self.llm_model,
                "llm_layers": self.llm_layers,
                "llm_dim": self.llm_dim,
                "patch_len": self.patch_len,
                "stride": self.stride,
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "n_heads": self.n_heads,
                "dropout": self.dropout,
                "enc_in": y.shape[1],
                "prompt_domain": self.prompt_domain,
            },
        ).load_model()

        self.model_ = self.model_.to(self.device_)
        self.model_ = self.model_.to(torch.bfloat16)

        self.last_values = y

    def _get_unique_time_llm_key(self):
        """Get unique key for Time-LLM model to use in multiton."""
        config_dict = {
            "task_name": self.task_name,
            "pred_len": self._pred_len,
            "seq_len": self.seq_len,
            "llm_model": self.llm_model,
            "llm_layers": self.llm_layers,
            "llm_dim": self.llm_dim,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "device": self.device,
            "prompt_domain": self.prompt_domain,
        }
        return str(sorted(config_dict.items()))

    def _predict(self, fh, X=None, y=None):
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
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
            (for_global)
            If ``y`` is not passed (not performing global forecasting), ``X`` should
            only contain the time points to be predicted.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.
        y : sktime time series object, optional (default=None) (for_global)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.


        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions
        """
        X_tensor = (
            torch.tensor(self.last_values.values).reshape(1, -1, 1).to(self.device_)
        )
        X_tensor = X_tensor.to(torch.float32)
        res = self.model_.forward(
            X_tensor, x_mark_enc=None, x_mark_dec=None, x_dec=None
        )

        forecast_index = fh.to_absolute(self.cutoff).to_pandas()

        y_pred = pd.DataFrame(
            data=res.detach().cpu().numpy().flatten(),
            index=forecast_index,
            columns=self.last_values.columns,
        )

        y_pred = y_pred.astype("float64")

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params_list = [
            {
                "task_name": "long_term_forecast",
                "pred_len": 24,
                "seq_len": 96,
                "llm_model": "GPT2",
                "llm_layers": 3,
                "llm_dim": 768,
                "patch_len": 16,
                "stride": 8,
                "d_model": 128,
                "d_ff": 128,
                "n_heads": 4,
                "dropout": 0.1,
                "device": None,
                "prompt_domain": False,
            },
            {
                "task_name": "short_term_forecast",
                "pred_len": 24,
                "seq_len": 96,
                "llm_model": "GPT2",
                "llm_layers": 3,
                "llm_dim": 768,
                "patch_len": 16,
                "stride": 8,
                "d_model": 128,
                "d_ff": 128,
                "n_heads": 4,
                "dropout": 0.1,
                "device": None,
                "prompt_domain": False,
            },
            {
                "task_name": "short_term_forecast",
                "pred_len": 24,
                "seq_len": 96,
                "llm_model": "GPT2",
                "llm_layers": 3,
                "llm_dim": 768,
                "patch_len": 16,
                "stride": 8,
                "d_model": 128,
                "d_ff": 128,
                "n_heads": 4,
                "dropout": 0.1,
                "device": None,
                "prompt_domain": False,
            },
        ]

        return params_list


@_multiton
class _CachedTimeLLM:
    """Cached Time-LLM model to ensure only one instance per configuration exists.

    Time-LLM is immutable and hence multiple instances with the same config can
    share the same model without any side effects.
    """

    def __init__(self, key, time_llm_kwargs):
        self.key = key
        self.time_llm_kwargs = time_llm_kwargs
        self.model_pipeline = None

    def load_model(self):
        """Load Time-LLM model from checkpoint if not already loaded."""
        if self.model_pipeline is not None:
            return self.model_pipeline

        from sktime.libs.time_llm.TimeLLM import Model

        configs = SimpleNamespace(**self.time_llm_kwargs)
        self.model_pipeline = Model(configs)
        return self.model_pipeline
