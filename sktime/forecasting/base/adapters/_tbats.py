# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for using tbats forecasters in sktime framework."""

__author__ = ["mloning", "aiwalter"]
__all__ = ["_TbatsAdapter"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.forecasting import check_sp


class _TbatsAdapter(BaseForecaster):
    """Base class for interfacing tbats forecasting algorithms."""

    _tags = {
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(
        self,
        use_box_cox=None,
        box_cox_bounds=(0, 1),
        use_trend=None,
        use_damped_trend=None,
        sp=None,
        use_arma_errors=True,
        show_warnings=True,
        n_jobs=None,
        multiprocessing_start_method="spawn",
        context=None,
    ):

        self.use_box_cox = use_box_cox
        self.box_cox_bounds = box_cox_bounds
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.sp = sp
        self.use_arma_errors = use_arma_errors
        self.show_warnings = show_warnings
        self.n_jobs = n_jobs
        self.multiprocessing_start_method = multiprocessing_start_method
        self.context = context
        # custom sktime args
        self._forecaster = None

        super(_TbatsAdapter, self).__init__()

    def _instantiate_model(self):
        n_jobs = check_n_jobs(self.n_jobs)
        sp = check_sp(self.sp, enforce_list=True)

        return self._ModelClass(
            use_box_cox=self.use_box_cox,
            box_cox_bounds=self.box_cox_bounds,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            seasonal_periods=sp,
            use_arma_errors=self.use_arma_errors,
            show_warnings=self.show_warnings,
            n_jobs=n_jobs,
            multiprocessing_start_method=self.multiprocessing_start_method,
            context=self.context,
        )

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (ignored)

        Returns
        -------
        self : returns an instance of self.
        """
        self._forecaster = self._instantiate_model()
        self._forecaster = self._forecaster.fit(y)

        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """
        fh = fh.to_relative(cutoff=self.cutoff)

        if not fh.is_all_in_sample(cutoff=self.cutoff):
            fh_out = fh.to_out_of_sample(cutoff=self.cutoff)
            steps = fh_out.to_pandas().max()
            out = pd.DataFrame(
                self._forecaster.forecast(steps=steps, confidence_level=1 - alpha)[1]
            )
            y_out = out["mean"]
            # pred_int
            lower = pd.Series(out["lower_bound"])
            upper = pd.Series(out["upper_bound"])
            pred_int = self._get_pred_int(lower=lower, upper=upper)

        else:
            y_out = np.array([])

        # y_pred
        y_in_sample = pd.Series(self._forecaster.y_hat)
        y_out_sample = pd.Series(y_out)
        y_pred = self._get_y_pred(y_in_sample=y_in_sample, y_out_sample=y_out_sample)

        if return_pred_int:
            return y_pred, pred_int
        else:
            return y_pred

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        self.check_is_fitted()
        fitted_params = {}
        for name in self._get_fitted_param_names():
            fitted_params[name] = getattr(self._forecaster, name, None)
        return fitted_params

    def _get_fitted_param_names(self):
        """Get names of fitted parameters."""
        return self._fitted_param_names

    def _get_pred_int(self, lower, upper):
        """Combine lower/upper bounds of pred.intervals, slice on fh.

        Parameters
        ----------
        lower : pd.Series
            Lower bound (can contain also in-sample bound)
        upper : pd.Series
            Upper bound (can contain also in-sample bound)

        Returns
        -------
        pd.DataFrame
            pred_int, predicion intervalls (out-sample, sliced by fh)
        """
        pred_int = pd.DataFrame({"lower": lower, "upper": upper})
        # Out-sample fh
        fh_out = self.fh.to_out_of_sample(cutoff=self.cutoff)
        # If pred_int contains in-sample prediction intervals
        if len(pred_int) > len(self._y):
            len_out = len(pred_int) - len(self._y)
            # Workaround for slicing with negative index
            pred_int["idx"] = [x for x in range(-len(self._y), len_out)]
        # If pred_int does not contain in-sample prediction intervals
        else:
            pred_int["idx"] = [x for x in range(len(pred_int))]
        pred_int = pred_int.loc[
            pred_int["idx"].isin(fh_out.to_indexer(self.cutoff).values)
        ]
        pred_int.index = fh_out.to_absolute(self.cutoff)
        pred_int = pred_int.drop(columns=["idx"])
        return pred_int
