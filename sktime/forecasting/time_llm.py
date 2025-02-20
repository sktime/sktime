"""Implements Time-LLM forecaster."""

__all__ = ["TimeLLMForecaster"]
__author__ = ["KimMeen", "jgyasu"]
# KimMeen for [ICLR 2024] Official implementation of Time-LLM

from types import SimpleNamespace
from typing import Optional

from sktime.forecasting.base import _BaseGlobalForecaster
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")


class TimeLLMForecaster(_BaseGlobalForecaster):
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
    ...     pred_len=36,
    ...     seq_len=96,
    ...     llm_model='GPT2'
    ... )
    >>> forecaster.fit(y_train)
    >>> y_pred = forecaster.predict(fh)
    """

    _tags = {
        "scitype:y": "univariate",
        "authors": ["KimMeen", "jgyasu"],
        # KimMeen for [ICLR 2024] Official implementation of Time-LLM
        "maintainers": ["jgyasu"],
        "python_dependencies": ["torch", "transformers"],
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "capability:global_forecasting": True,
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
        broadcasting=False,
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
        self.broadcasting = broadcasting
        self.prompt_domain = prompt_domain

        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.Series",
                    "X_inner_mtype": "pd.DataFrame",
                    "capability:global_forecasting": False,
                }
            )

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.DataFrame
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here.
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        configs = SimpleNamespace(
            task_name=self.task_name,
            pred_len=self.pred_len,
            seq_len=self.seq_len,
            llm_model=self.llm_model,
            llm_layers=self.llm_layers,
            llm_dim=self.llm_dim,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            dropout=self.dropout,
            enc_in=1 if self.broadcasting else y.shape[1],
            prompt_domain=False,
        )

        if self.device is None:
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_ = self.device

        from sktime.libs.time_llm.TimeLLM import Model

        self.model_ = Model(configs).to(self.device_)

        # todo

        return self

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
        # todo

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
                "llm_model": "BERT",
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
