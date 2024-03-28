# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Chronos forecaster by wrapping amazon's chronos."""

__author__ = ["RigvedManoj"]
__all__ = ["Chronos"]


import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class Chronos(BaseForecaster):
    """Chronos forecaster by wrapping Amazon's Chronos model [1]_.

    Direct interface to Amazon Chronos, using the sktime interface.
    All hyperparameters are exposed via the constructor.

    Parameters
    ----------
    model_name: str, required
    top_p: float, default=1.0
    top_k: int, default=50
    temperature: float, default=1.0
    num_samples: int, default=20
    args_list: list, default=None
    kwargs_dict: dict, default=None

    References
    ----------
    . [1] https://github.com/amazon-science/chronos-forecasting

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.chronos import Chronos
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> import torch
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> forecaster = Chronos(
    ...        "amazon/chronos-t5-small",
    ...        kwargs_dict={"torch_dtype": torch.bfloat16}
        )  # doctest: +SKIP
    >>> forecaster.fit(y_train)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh) # doctest: +SKIP
    """

    # tag values are "safe defaults" which can usually be left as-is
    _tags = {
        "python_dependencies": ["torch"],
        "y_inner_mtype": "pd.Series",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "authors": ["RigvedManoj"],
        "maintainers": ["RigvedManoj"],
    }

    def __init__(
        self,
        model_name,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        args_list=None,
        kwargs_dict=None,
    ):
        self.model_name = model_name
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.args_list = args_list
        self.kwargs_dict = kwargs_dict

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

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
        import torch
        from chronos import ChronosPipeline

        if self.args_list is not None:
            args_list = [self.model_name] + self.args_list
        else:
            args_list = [self.model_name]
        if self.kwargs_dict is None:
            self._model = ChronosPipeline.from_pretrained(*args_list)
        else:
            self._model = ChronosPipeline.from_pretrained(
                *args_list, **self.kwargs_dict
            )
        self._context = torch.tensor(y.values)
        return self

    def _predict(self, fh=None, X=None):
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
        prediction_length = len(fh)
        forecast = self._model.predict(
            self._context,
            prediction_length,
            num_samples=self.num_samples,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        values = np.median(forecast[0].numpy(), axis=0)
        """
        row_idx = fh.to_absolute_index(self.cutoff)
        col_idx = self._y.index
        """
        y_pred = pd.DataFrame(values)
        return y_pred

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
        params = {"model_name": "amazon/chronos-t5-small"}
        return params
