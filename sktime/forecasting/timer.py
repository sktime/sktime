# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interface for the Timer foundation model for time series forecasting.

Timer is a generative pre-trained Transformer for time series, developed by
the THUML group at Tsinghua University. It treats forecasting, imputation,
and anomaly detection as a unified generative task.

References
----------
.. [1] Liu et al., "Timer: Generative Pre-trained Transformers Are Large
   Time Series Models", ICML 2024.
.. [2] Liu et al., "Timer-XL: Long-Context Transformers for Unified Time
   Series Forecasting", ICLR 2025.
"""

__author__ = ["PewterZz"]
__all__ = ["TimerForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class TimerForecaster(BaseForecaster):
    """Timer foundation model forecaster.

    Wraps the Timer generative pre-trained Transformer for zero-shot
    time series forecasting via the HuggingFace ``transformers`` library.

    Timer uses autoregressive generation on continuous time series tokens.
    The model is pre-trained on the Unified Time Series Dataset (UTSD)
    covering diverse domains and temporal patterns.

    Parameters
    ----------
    model_name : str, default="thuml/timer-base-84m"
        Name or path of the pre-trained Timer model on HuggingFace.
        Options include:

        - "thuml/timer-base-84m" (84M parameters)
        - "thuml/timer-xl-84m" (Timer-XL variant)

    context_length : int, default=2880
        Number of historical observations to use as input context.
        Timer supports variable context lengths. If the series is shorter,
        the full series is used.
    device : str, default="cpu"
        Device to run the model on. Options: "cpu", "cuda", "cuda:0", etc.

    References
    ----------
    .. [1] Liu et al., "Timer: Generative Pre-trained Transformers Are Large
       Time Series Models", ICML 2024.
       https://arxiv.org/abs/2402.02368
    .. [2] Liu et al., "Timer-XL: Long-Context Transformers for Unified Time
       Series Forecasting", ICLR 2025.

    Examples
    --------
    >>> from sktime.forecasting.timer import TimerForecaster  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = TimerForecaster(
    ...     model_name="thuml/timer-base-84m",
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    # todo: consider adding _multiton caching pattern (as in ChronosForecaster)
    #  to avoid reloading the model when multiple instances share the same weights.
    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "python_dependencies": ["transformers>=4.40", "torch"],
    }

    def __init__(
        self,
        model_name="thuml/timer-base-84m",
        context_length=2880,
        device="cpu",
    ):
        self.model_name = model_name
        self.context_length = context_length
        self.device = device

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Loads the pre-trained Timer model and stores the training series
        as context for prediction. The Timer model is pre-trained and does
        not require task-specific training for zero-shot forecasting.

        Parameters
        ----------
        y : pd.Series
            Training time series.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables. Ignored.
        fh : ForecastingHorizon, optional (default=None)
            Forecasting horizon.

        Returns
        -------
        self
        """
        import torch
        from transformers import AutoModelForCausalLM

        self._y_train = y.values.astype(np.float32)

        self.model_ = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model_.to(self.device)
        self.model_.eval()

        return self

    def _predict(self, fh, X=None):
        """Make forecasts for the given forecasting horizon.

        Uses autoregressive generation to produce forecasts. Timer generates
        continuous-valued time series tokens, not discrete tokens.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon, relative or absolute.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables. Ignored.

        Returns
        -------
        y_pred : pd.Series
            Forecasted values indexed by the forecasting horizon.
        """
        import torch

        fh_relative = fh.to_relative(self.cutoff)
        max_h = max(fh_relative)

        # Prepare context: use most recent context_length observations
        context = self._y_train
        if len(context) > self.context_length:
            context = context[-self.context_length :]

        # Timer expects shape (batch_size, seq_len)
        input_tensor = torch.tensor(
            context, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            output = self.model_.generate(
                input_tensor, max_new_tokens=max_h
            )

        # output shape: (batch_size, max_h) -- Timer returns only the forecast
        forecast_values = output[0].cpu().numpy()

        # Select only the requested horizon indices
        fh_idx = np.array(fh_relative) - 1  # convert to 0-indexed
        valid_idx = fh_idx[fh_idx < len(forecast_values)]

        if len(valid_idx) < len(fh_relative):
            y_pred_values = np.full(len(fh_relative), np.nan)
            y_pred_values[: len(valid_idx)] = forecast_values[valid_idx]
        else:
            y_pred_values = forecast_values[fh_idx]

        # Build output index
        fh_abs = fh.to_absolute(self.cutoff)
        index = fh_abs.to_pandas()

        return pd.Series(y_pred_values, index=index, name=self._y.name)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the testing parameter set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {
            "model_name": "thuml/timer-base-84m",
            "context_length": 512,
            "device": "cpu",
        }
        return [params1]
