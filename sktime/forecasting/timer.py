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

from sktime.forecasting.foundation._base2 import BaseFoundationForecaster
from sktime.forecasting.foundation._result import ForecastResult, ModelHandle


class TimerForecaster(BaseFoundationForecaster):
    """Timer foundation model forecaster.

    Wraps the Timer generative pre-trained Transformer for zero-shot
    time series forecasting via the HuggingFace ``transformers`` library.

    Timer uses autoregressive generation on continuous time series tokens.
    The model is pre-trained on the Unified Time Series Dataset (UTSD)
    covering diverse domains and temporal patterns.

    The model is cached by the shared foundation-model lifecycle to avoid
    reloading weights when multiple forecaster instances share the same model.

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

    _tags = {
        # packaging info
        # --------------
        "authors": ["PewterZz"],
        "maintainers": "PewterZz",
        "python_dependencies": [
            "transformers>=4.40,<4.41",
            "torch",
        ],
        "python_version": "<3.13",
        # estimator type
        # --------------
        "capability:multivariate": False,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:insample": False,
        # CI and test tags
        # ----------------
        "tests:vm": True,
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

        super().__init__(model_path=model_name, device=device)

    def _load_model(self):
        """Load a Timer checkpoint into a cacheable model handle."""
        from sktime.libs.timer import TimerForPrediction

        model = TimerForPrediction.from_pretrained(self.model_path)
        model = model.to(self.device_)
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
        """Run Timer autoregressive generation and normalize its output."""
        import torch

        model = handle.model
        context = context_y.iloc[:, 0].to_numpy(dtype=np.float32)

        # Timer requires at least one complete input token. Left-padding retains
        # the previous behavior for short series used in estimator checks.
        token_len = model.config.input_token_len
        if len(context) < token_len:
            context = np.pad(context, (token_len - len(context), 0))

        if len(context) > self.context_length:
            context = context[-self.context_length :]

        # Timer requires input length to be a multiple of input_token_len
        usable_len = (len(context) // token_len) * token_len
        context = context[-usable_len:]

        # Timer expects shape (batch_size, seq_len)
        input_tensor = torch.tensor(
            context, dtype=torch.float32, device=self.device_
        ).unsqueeze(0)

        output = model.generate(input_tensor, max_new_tokens=pred_len)

        # output shape: (batch_size, max_h) -- Timer returns only the forecast
        forecast_values = output[0].cpu().numpy()
        return ForecastResult(mean=forecast_values.reshape(-1, 1), raw=output)

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
            "context_length": 960,
            "device": "cpu",
        }
        params2 = {
            "model_name": "thuml/timer-base-84m",
            "context_length": 1920,
            "device": "cpu",
        }
        return [params1, params2]
