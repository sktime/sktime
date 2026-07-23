"""Implements Time-LLM forecaster."""

__all__ = ["TimeLLMForecaster"]
__author__ = ["KimMeen", "jgyasu"]
# KimMeen for [ICLR 2024] Official implementation of Time-LLM

from types import SimpleNamespace

from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    FoundationModelSpec,
    ModelHandle,
)
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")


class TimeLLMForecaster(BaseFoundationForecaster):
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
    TimeLLMForecaster(pred_len=36)
    >>> y_pred = forecaster.predict(fh=[1])
    """

    _tags = {
        "capability:multivariate": False,
        "authors": ["KimMeen", "jgyasu"],
        # KimMeen for [ICLR 2024] Official implementation of Time-LLM
        "maintainers": ["jgyasu"],
        "python_dependencies": ["torch", "transformers"],
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": False,
        "requires-fh-in-fit": True,
        # testing configuration
        # ---------------------
        "tests:vm": True,
        "tests:libs": ["sktime.libs.time_llm"],
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
        device: str | None = None,
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
        model_spec = FoundationModelSpec(
            device=device,
            dtype="torch.bfloat16",
            load_extra_kwargs={
                "task_name": task_name,
                "pred_len": pred_len,
                "seq_len": seq_len,
                "llm_model": llm_model,
                "llm_layers": llm_layers,
                "llm_dim": llm_dim,
                "patch_len": patch_len,
                "stride": stride,
                "d_model": d_model,
                "d_ff": d_ff,
                "n_heads": n_heads,
                "dropout": dropout,
                "prompt_domain": prompt_domain,
            },
        )
        super().__init__(model_spec=model_spec)

    def _update_attrs_in_fit(self, y, X=None, fh=None):
        """Set the data-dependent Time-LLM architecture parameters."""
        load_extra_kwargs = dict(self.model_spec.load_extra_kwargs)
        if fh is None:
            self._pred_len = load_extra_kwargs["pred_len"]
        else:
            relative_fh = fh.to_relative(self.cutoff).to_pandas()
            if all(value > 0 for value in relative_fh):
                self._pred_len = int(max(relative_fh))
            else:
                self._pred_len = len(relative_fh)

        self._enc_in = y.shape[1]
        load_extra_kwargs.update(pred_len=self._pred_len, enc_in=self._enc_in)
        self._update_model_spec(
            load_extra_kwargs=load_extra_kwargs,
        )

    def _load_model(self):
        """Construct Time-LLM and return its shared model handle."""
        from sktime.libs.time_llm.TimeLLM import Model

        model_spec = self.model_spec
        configs = SimpleNamespace(**model_spec.load_extra_kwargs)
        model = Model(configs)
        model = model.to(model_spec.device)
        model = model.to(model_spec.dtype)
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
        """Run Time-LLM inference and return normalized point forecasts."""
        device = self.model_spec.device
        X_tensor = torch.tensor(context_y.values).reshape(1, -1, 1).to(device)
        X_tensor = X_tensor.to(torch.float32)

        res = handle.model.forward(
            X_tensor, x_mark_enc=None, x_mark_dec=None, x_dec=None
        )
        values = res.detach().to(torch.float32).cpu().numpy().astype("float64")
        values = values.reshape(-1, context_y.shape[1])
        return ForecastResult(mean=values)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        minimal_params = {
            "pred_len": 1,
            "seq_len": 5,
            "llm_model": "TINY_RANDOM",
            "llm_layers": 1,
            "llm_dim": 1,
            "patch_len": 5,
            "stride": 5,
            "d_model": 1,
            "d_ff": 1,
            "n_heads": 1,
            "dropout": 0.0,
            "device": "cpu",
            "prompt_domain": False,
        }
        params_list = [
            {
                **minimal_params,
                "task_name": "long_term_forecast",
            },
            {
                **minimal_params,
                "task_name": "short_term_forecast",
            },
        ]

        return params_list
