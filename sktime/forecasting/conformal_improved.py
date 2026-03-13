# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Conformal prediction intervals for sktime forecasters."""

import numpy as np
import pandas as pd

from sktime.forecasting.base._base import BaseForecaster
from sktime.utils._mtype import MTYPE_LIST_SERIES


class ConformalIntervals(BaseForecaster):
    """Conformal prediction intervals for sktime forecasters.

    This estimator wraps another forecaster and adds conformal
    prediction intervals to the predictions. Conformal intervals are
    guaranteed to have the correct coverage probability under mild
    assumptions about the data distribution.

    Parameters
    ----------
    forecaster : sktime forecaster
        The base forecaster to wrap. The forecaster must implement
        `predict_intervals`.
    method : str, default="empirical"
        Method to use for computing conformal intervals. Supported
        methods are:
        - "empirical": empirical quantiles of residuals
        - "conformal": conformal prediction intervals
        - "conformal_bonferroni": Bonferroni corrected conformal intervals
    alpha : float, default=0.05
        Significance level for the prediction intervals. Default is 0.05
        for 95% confidence intervals.

    Examples
    --------
    Basic usage with NaiveForecaster:
    
    >>> from sktime.forecasting.conformal import ConformalIntervals
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="last")
    >>> conformal_forecaster = ConformalIntervals(forecaster, alpha=0.05)
    >>> conformal_forecaster.fit(y, fh=[1, 2, 3])
    >>> y_pred_intervals = conformal_forecaster.predict_intervals()
    >>> print(y_pred_intervals.head())
    
    Adding conformal intervals to a grid search:
    
    >>> from sktime.forecasting.model_selection import ForecastingGridSearchCV
    >>> from sktime.forecasting.model_evaluation import ExpandingWindowCV
    >>> cv = ExpandingWindowCV(forecasting_horizon=12)
    >>> forecaster = NaiveForecaster()
    >>> param_grid = {"strategy" : ["last", "mean", "drift"]}
    >>> gscv = ForecastingGridSearchCV(
    ...     forecaster=forecaster,
    ...     param_grid=param_grid,
    ...     cv=cv,
    ... )
    >>> conformal_with_grid = ConformalIntervals(gscv)
    >>> conformal_with_grid.fit(y, fh=[1, 2, 3])
    >>> y_pred_intervals = conformal_with_grid.predict_intervals()
    """

    _config = {
        "forecaster": None,
        "method": "empirical",
        "alpha": 0.05,
    }

    _tags = {
        "authors": ["fkiraly", "bethrice44"],
        "python_dependencies": ["joblib"],
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:exogenous": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "X_inner_mtype": MTYPE_LIST_SERIES,
        "y_inner_mtype": MTYPE_LIST_SERIES,
        "tests:core": True,
    }

    ALLOWED_METHODS = [
        "empirical",
        "empirical_residual",
        "conformal",
        "conformal_bonferroni",
    ]

    def __init__(
        self,
        forecaster,
        method="empirical",
        alpha=0.05,
    ):
        self.forecaster = forecaster
        self.method = method
        self.alpha = alpha
        super().__init__()

        if method not in self.ALLOWED_METHODS:
            raise ValueError(f"Method {method} is not supported.")

    def _fit(self, y, X=None, fh=None):
        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(y, X=X, fh=fh)
        return self

    def _predict(self, fh, X=None):
        return self.forecaster_.predict(fh, X=X)
