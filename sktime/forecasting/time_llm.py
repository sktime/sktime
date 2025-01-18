"""Implements Time-LLM forecaster."""

__all__ = ["TimeLLMForecaster"]
__author__ = ["KimMeen", "jgyasu"]
# KimMeen for [ICLR 2024] Official implementation of Time-LLM

from types import SimpleNamespace
from typing import Optional

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation.forecasting import check_X


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
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> forecaster = TimeLLMForecaster(
    ...     task_name="long_term_forecast",
    ...     llm_model="GPT2"
    ...)
    >>> forecaster.fit(y_train)
    >>> y_pred = forecaster.predict(fh)
    """

    _tags = {
        "scitype:y": "both",
        "authors": ["KimMeen", "jgyasu"],
        # KimMeen for [ICLR 2024] Official implementation of Time-LLM
        "maintainers": ["jgyasu"],
        "python_dependencies": ["torch==2.2.2", "transformers==4.31.0"],
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": True,
    }

    def __init__(
        self,
        task_name="long_term_forecast",
        pred_len=24,
        seq_len=96,
        llm_model="GPT2",
        llm_layers=3,
        patch_len=16,
        stride=8,
        d_model=128,
        d_ff=128,
        n_heads=4,
        dropout=0.1,
        device: Optional[str] = None,
    ):
        self.task_name = task_name
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.llm_model = llm_model
        self.llm_layers = llm_layers
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = dropout

        self.device = device

        super().__init__()

    def _model_config(self, n_variables):
        return SimpleNamespace(
            task_name=self.task_name,
            pred_len=self.pred_len,
            seq_len=self.seq_len,
            enc_in=n_variables,
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            llm_layers=self.llm_layers,
            dropout=self.dropout,
            patch_len=self.patch_len,
            stride=self.stride,
            llm_model=self.llm_model,
            llm_dim=768 if self.llm_model in ["GPT2", "BERT"] else 4096,
            prompt_domain=False,
        )

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        if _check_soft_dependencies("torch", severity="none"):
            import torch
        from sktime.libs.time_llm.TimeLLM import Model

        X = check_X(X)

        if isinstance(X, pd.DataFrame):
            self.n_variables_ = X.shape[1]
        else:
            self.n_variables_ = 1

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device)

        config = self._model_config(self.n_variables_)
        self.model_ = Model(config).to(self.device)

        self._is_fitted = True
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon."""
        if _check_soft_dependencies("torch", severity="none"):
            import torch

        if not self._is_fitted:
            raise RuntimeError("Forecaster is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            x_enc = torch.FloatTensor(X.values).unsqueeze(0)
        else:
            x_enc = torch.FloatTensor(X).unsqueeze(0).unsqueeze(-1)

        x_enc = x_enc.to(self.device)

        x_mark_enc = torch.zeros((1, x_enc.shape[1], 1)).to(self.device)
        x_dec = torch.zeros((1, self.pred_len, x_enc.shape[2])).to(self.device)
        x_mark_dec = torch.zeros((1, self.pred_len, 1)).to(self.device)

        with torch.no_grad():
            predictions = self.model_(x_enc, x_mark_enc, x_dec, x_mark_dec)

        predictions = predictions.cpu().numpy().squeeze()

        if self.n_variables_ == 1:
            predictions = predictions.reshape(-1, 1)

        index = pd.RangeIndex(
            start=X.index[-1] + 1, stop=X.index[-1] + self.pred_len + 1
        )

        if isinstance(X, pd.DataFrame):
            columns = X.columns
            predictions = pd.DataFrame(predictions, index=index, columns=columns)
        else:
            predictions = pd.DataFrame(predictions, index=index)

        return predictions

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
                "patch_len": 16,
                "stride": 8,
                "d_model": 128,
                "d_ff": 128,
                "n_heads": 4,
                "dropout": 0.1,
                "device": None,
            },
            {
                "task_name": "short_term_forecast",
                "pred_len": 24,
                "seq_len": 96,
                "llm_model": "BERT",
                "llm_layers": 3,
                "patch_len": 16,
                "stride": 8,
                "d_model": 128,
                "d_ff": 128,
                "n_heads": 4,
                "dropout": 0.1,
                "device": None,
            },
            {
                "task_name": "long_term_forecast",
                "pred_len": 24,
                "seq_len": 96,
                "llm_model": "LLAMA",
                "llm_layers": 3,
                "patch_len": 16,
                "stride": 8,
                "d_model": 128,
                "d_ff": 128,
                "n_heads": 4,
                "dropout": 0.1,
                "device": None,
            },
        ]

        return params_list
