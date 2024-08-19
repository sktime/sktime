"""Implements Chronos forecaster."""

__author__ = ["Z-Fran"]
__all__ = ["ChronosForecaster"]

from typing import Optional

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.libs.chronos import ChronosPipeline

if _check_soft_dependencies("torch", severity="none"):
    import torch

if _check_soft_dependencies("transformers", severity="none"):
    import transformers


class ChronosForecaster(BaseForecaster):
    """Chronos forecaster.

    Parameters
    ----------
    model_path : str
        Path to the Chronos' huggingface model.
    config : dict, default={}
        Configuration to use for the model.
    seed: int, optional, default=None
        Random seed for transformers.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.chronos import ChronosForecaster
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> forecaster = ChronosForecaster("amazon/chronos-t5-tiny")
    >>> forecaster.fit(y_train)
    >>> y_pred = forecaster.predict(fh)
    """

    # tag values are "safe defaults" which can usually be left as-is
    _tags = {
        "python_dependencies": ["torch", "transformers"],
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.Series",
        "scitype:y": "univariate",
        "authors": ["Z-Fran"],
    }

    _default_config = {
        "num_samples": None,  # int, use value from pretrained model if None
        "temperature": None,  # float, use value from pretrained model if None
        "top_k": None,  # int, use value from pretrained model if None
        "top_p": None,  # float, use value from pretrained model if None
        "limit_prediction_length": False,  # bool
        "torch_dtype": torch.bfloat16,  # torch.dtype
        "device_map": "cpu",  # str, use "cpu" for CPU inference, "cuda" for gpu and "mps" for Apple Silicon # noqa
    }

    def __init__(
        self,
        model_path: str,
        config: dict = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # set random seed
        self.seed = seed
        self._seed = np.random.randint(0, 2**31) if seed is None else seed

        # set config
        self.config = config
        _config = self._default_config.copy()
        _config.update(config if config is not None else {})
        self._config = _config

        self.model_path = model_path
        self.model_pipeline = None
        self.context = None

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        self : reference to self
        """
        if self.model_pipeline is None:
            self.model_pipeline = ChronosPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self._config["torch_dtype"],
                device_map=self._config["device_map"],
            )
        self.context = torch.tensor(y.values)
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        y_pred : pd.DataFrame
            Predicted forecasts.
        """
        transformers.set_seed(self._seed)
        prediction_length = len(fh)

        prediction_results = self.model_pipeline.predict(
            self.context,
            prediction_length,
            num_samples=self._config["num_samples"],
            temperature=self._config["temperature"],
            top_k=self._config["top_k"],
            top_p=self._config["top_p"],
        )

        values = np.median(prediction_results[0].numpy(), axis=0)
        row_idx = self.fh.to_absolute_index(self.cutoff)
        y_pred = pd.Series(
            values, index=row_idx, name=self._y.name if self._y is not None else None
        )
        return y_pred

    def __repr__(self):
        """Repr dunder."""
        class_name = self.__class__.__name__
        return f"{class_name}"

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        params = []
        params.append(
            {
                "model_path": "amazon/chronos-t5-tiny",
            }
        )
        params.append(
            {
                "model_path": "amazon/chronos-t5-samll",
                "config": {
                    "num_samples": 20,
                },
                "seed": 42,
            }
        )

        return params
