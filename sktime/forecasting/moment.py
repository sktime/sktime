# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements MOMENT foundation model forecaster.

MOMENT is a family of open-source foundation models for general-purpose
time series analysis, developed by the Auton Lab at Carnegie Mellon University.

Reference
---------
.. [1] Goswami et al., "MOMENT: A Family of Open Time-Series Foundation Models",
   ICML 2024. https://arxiv.org/abs/2402.03885
"""

__author__ = ["PewterZz"]
__all__ = ["MomentForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class MomentForecaster(BaseForecaster):
    """MOMENT foundation model forecaster.

    Wraps the MOMENT time series foundation model for zero-shot and
    fine-tuned forecasting. MOMENT is a multi-task foundation model
    supporting forecasting, classification, anomaly detection, and
    imputation. This wrapper exposes the forecasting capability.

    MOMENT uses a pre-trained transformer architecture that operates on
    fixed-length input patches. The model is available in Small, Base,
    and Large variants via HuggingFace.

    Parameters
    ----------
    model_name : str, default="AutonLab/MOMENT-1-large"
        Name or path of the pre-trained MOMENT model on HuggingFace.
        Options include:

        - "AutonLab/MOMENT-1-small"
        - "AutonLab/MOMENT-1-base"
        - "AutonLab/MOMENT-1-large"

    input_length : int, default=512
        Length of the input time series window. MOMENT uses fixed-length
        input patches. If the series is shorter, it will be padded.
        If longer, the most recent ``input_length`` observations are used.
    device : str, default="cpu"
        Device to run the model on. Options: "cpu", "cuda", "cuda:0", etc.

    References
    ----------
    .. [1] Goswami et al., "MOMENT: A Family of Open Time-Series Foundation
       Models", ICML 2024. https://arxiv.org/abs/2402.03885

    Examples
    --------
    >>> from sktime.forecasting.moment import MomentForecaster  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = MomentForecaster(
    ...     model_name="AutonLab/MOMENT-1-large",
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3, 4, 5])  # doctest: +SKIP
    """

    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "python_dependencies": ["momentfm", "torch"],
        "python_version": ">=3.10",
    }

    def __init__(
        self,
        model_name="AutonLab/MOMENT-1-large",
        input_length=512,
        device="cpu",
    ):
        self.model_name = model_name
        self.input_length = input_length
        self.device = device

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Stores the training series for use as context during prediction.
        The MOMENT model itself is pre-trained and does not require fitting,
        but the training data is needed as input context for forecasting.

        Parameters
        ----------
        y : pd.Series
            Training time series.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables. Ignored by MOMENT.
        fh : ForecastingHorizon, optional (default=None)
            Forecasting horizon.

        Returns
        -------
        self
        """
        from momentfm import MOMENTPipeline

        self._y_train = y.values.astype(np.float32)

        # Determine forecast horizon for model initialization
        if fh is not None:
            self._forecast_horizon = max(fh.to_relative(self.cutoff))
        else:
            self._forecast_horizon = 1

        self.model_ = MOMENTPipeline.from_pretrained(
            self.model_name,
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": self._forecast_horizon,
            },
        )
        self.model_.init()

        return self

    def _predict(self, fh, X=None):
        """Make forecasts for the given forecasting horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon, relative or absolute.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables. Ignored by MOMENT.

        Returns
        -------
        y_pred : pd.Series
            Forecasted values indexed by the forecasting horizon.
        """
        import torch

        fh_relative = fh.to_relative(self.cutoff)
        max_h = max(fh_relative)

        # Re-initialize model if forecast horizon changed since fit
        if max_h != self._forecast_horizon:
            from momentfm import MOMENTPipeline

            self._forecast_horizon = max_h
            self.model_ = MOMENTPipeline.from_pretrained(
                self.model_name,
                model_kwargs={
                    "task_name": "forecasting",
                    "forecast_horizon": self._forecast_horizon,
                },
            )
            self.model_.init()

        # Prepare input: use the most recent input_length observations
        context = self._y_train
        if len(context) > self.input_length:
            context = context[-self.input_length :]

        # MOMENT expects shape (batch, n_channels, seq_len)
        input_tensor = torch.tensor(context, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.model_(input_tensor)

        # Extract forecast from model output
        forecast_values = output.forecast.squeeze().numpy()

        # Select only the requested horizon indices
        fh_idx = np.array(fh_relative) - 1  # 0-indexed
        fh_idx = fh_idx[fh_idx < len(forecast_values)]

        if len(fh_idx) < len(fh_relative):
            # Pad with last value if requested horizon exceeds model output
            padded = np.full(len(fh_relative), forecast_values[-1])
            padded[: len(fh_idx)] = forecast_values[fh_idx]
            y_pred_values = padded
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
        params : dict
            Parameters to create testing instances of the class.
        """
        params1 = {
            "model_name": "AutonLab/MOMENT-1-small",
            "input_length": 64,
            "device": "cpu",
        }
        return [params1]
