# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements TSICLForecaster — sktime wrapper for TS-ICL foundation model."""

__author__ = ["amruth6002"]
__all__ = ["TSICLForecaster"]

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


@_multiton
class _CachedTSICL:
    """Cache wrapper for the TSICL model."""

    def __init__(self, key, tsicl_kwargs):
        from tsicl.pipeline import TSICL

        self._model = TSICL(**tsicl_kwargs)

    @property
    def model(self):
        return self._model


class TSICLForecaster(BaseForecaster):
    """sktime interface to TS-ICL (Time Series In-Context Learning) foundation model.

    TS-ICL is a zero-shot, in-context learning foundation model for time series
    forecasting developed by EDF-Lab [1]_. It uses an encoder-decoder architecture
    with Perceiver-style cross-attention to handle arbitrary context lengths,
    multi-horizon forecasts, and missing values natively without retraining.

    Parameters
    ----------
    model_path : str
        HuggingFace model repository ID (e.g., "taharnbl/TS-ICL") or path
        to a local model checkpoint.
    quantile_levels : list of float, default=None
        Quantile levels to predict during probabilistic forecasting.
        Values must be between 0.0 and 1.0. If None, defaults to [0.1, 0.5, 0.9].
    denormalize : bool, default=True
        Whether to re-scale the forecasts back to the original input data scale.

    Attributes
    ----------
    model_ : tsicl.pipeline.TSICL
        Fitted TS-ICL pipeline instance retrieved from the multiton cache.

    References
    ----------
    .. [1] Tahar Allouche et al. "TS-ICL: In-Context Learning for Zero-Shot
       Time Series Forecasting", EDF-Lab. https://github.com/EDF-Lab/ts-icl

    Examples
    --------
    >>> from sktime.forecasting.tsicl import TSICLForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = TSICLForecaster(model_path="taharnbl/TS-ICL")  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "authors": ["amruth6002"],
        "maintainers": ["amruth6002"],
        "python_version": ">=3.12",
        "python_dependencies": ["tsicl", "torch"],
        "capability:exogenous": False,
        "capability:missing_values": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:insample": False,
        "capability:pretrain": False,
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path: str,
        quantile_levels: list = None,
        denormalize: bool = True,
    ):
        self.model_path = model_path
        self.quantile_levels = quantile_levels
        self.denormalize = denormalize

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        TS-ICL is a zero-shot foundation model, so fitting stores the
        historical context series and loads the model checkpoint into memory.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series data to store as context history.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series (currently unused).
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        self._y_train = y
        self.model_ = self._load_model()

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon with steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series (currently unused).

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions for the requested forecasting horizon.
        """
        model = self._load_model()
        y_values = self._y_train.iloc[:, 0].to_numpy()

        prediction_length = max(fh.to_relative(self.cutoff))

        point, _ = model.forecast(
            inputs=y_values,
            prediction_length=prediction_length,
            denormalize=self.denormalize,
            quantile_levels=self.quantile_levels,
        )

        fh_relative = fh.to_relative(self.cutoff)
        fh_idx = [int(i) - 1 for i in fh_relative]
        point_selected = point[fh_idx]

        row_idx = fh.to_absolute_index(self.cutoff)
        col_idx = self._get_varnames()
        y_pred = pd.DataFrame(
            point_selected.reshape(-1, 1), index=row_idx, columns=col_idx
        )
        return y_pred

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute and return prediction quantiles for the forecast.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon with steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series (currently unused).
        alpha : list of float, optional (default=None)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            MultiIndex DataFrame with variable name at level 0 and
            quantile probabilities (alpha) at level 1.
        """
        if alpha is None:
            alpha = (
                self.quantile_levels
                if self.quantile_levels is not None
                else [0.1, 0.5, 0.9]
            )
        model = self._load_model()
        y_values = self._y_train.iloc[:, 0].to_numpy()
        prediction_length = max(fh.to_relative(self.cutoff))
        _, quantiles = model.forecast(
            inputs=y_values,
            prediction_length=prediction_length,
            quantile_levels=alpha,
            denormalize=self.denormalize,
        )

        fh_relative = fh.to_relative(self.cutoff)
        fh_idx = [int(i) - 1 for i in fh_relative]
        quantiles_selected = quantiles[fh_idx]

        row_idx = fh.to_absolute_index(self.cutoff)
        var_name = self._get_varnames()[0]

        col_idx = pd.MultiIndex.from_product([[var_name], alpha])
        result = pd.DataFrame(quantiles_selected, index=row_idx, columns=col_idx)
        return result

    def __getstate__(self):
        """Return state for pickling, excluding unpickleable model."""
        state = self.__dict__.copy()
        if hasattr(self, "model_"):
            state["model_"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled state dictionary."""
        self.__dict__.update(state)

    def _load_model(self):
        """Reload the model from cache if it was dropped during pickling."""
        if getattr(self, "model_", None) is None:
            self.model_ = _CachedTSICL(
                key=self.model_path,
                tsicl_kwargs={"model_path": self.model_path},
            ).model
        return self.model_

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set to return.

        Returns
        -------
        params : list of dict
            List of parameter settings to test.
        """
        params1 = {
            "model_path": "taharnbl/TS-ICL",
            "quantile_levels": [0.1, 0.5, 0.9],
            "denormalize": True,
        }
        params2 = {
            "model_path": "taharnbl/TS-ICL",
            "quantile_levels": [0.25, 0.75],
            "denormalize": False,
        }
        return [params1, params2]
