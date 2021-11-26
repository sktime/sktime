# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for pmdarima forecasters to be used in sktime framework."""

__author__ = ["Markus Löning", "Hongyi Yang"]
__all__ = ["_PmdArimaAdapter"]

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA


class _PmdArimaAdapter(BaseForecaster):
    """Base class for interfacing pmdarima."""

    _tags = {
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self):
        self._forecaster = None
        super(_PmdArimaAdapter, self).__init__()

    def _instantiate_model(self):
        raise NotImplementedError("abstract method")

    def _fit(self, y, X=None, fh=None, **fit_params):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        self._forecaster = self._instantiate_model()
        self._forecaster.fit(y, X=X, **fit_params)
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        # distinguish between in-sample and out-of-sample prediction
        fh_oos = fh.to_out_of_sample(self.cutoff)
        fh_ins = fh.to_in_sample(self.cutoff)

        kwargs = {"X": X, "return_pred_int": return_pred_int, "alpha": alpha}

        # all values are out-of-sample
        if fh.is_all_out_of_sample(self.cutoff):
            return self._predict_fixed_cutoff(fh_oos, **kwargs)

        # all values are in-sample
        elif fh.is_all_in_sample(self.cutoff):
            return self._predict_in_sample(fh_ins, **kwargs)

        # both in-sample and out-of-sample values
        else:
            if return_pred_int:
                y_ins_pred, y_ins_pred_int = self._predict_in_sample(fh_ins, **kwargs)
                y_oos_pred, y_oos_pred_int = self._predict_fixed_cutoff(
                    fh_oos, **kwargs
                )
                return y_ins_pred.append(y_oos_pred), y_ins_pred_int.append(
                    y_oos_pred_int
                )
            else:
                y_ins = self._predict_in_sample(fh_ins, **kwargs)
                y_oos = self._predict_fixed_cutoff(fh_oos, **kwargs)
                return y_ins.append(y_oos)

    def _predict_in_sample(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        if isinstance(alpha, (list, tuple)):
            raise NotImplementedError("multiple `alpha` values are not yet supported")

        if hasattr(self, "order"):
            diff_order = self.order[1]
        else:
            diff_order = self._forecaster.model_.order[1]

        # Initialize return objects
        fh_abs = fh.to_absolute(self.cutoff).to_numpy()
        fh_idx = fh.to_indexer(self.cutoff, from_cutoff=False)
        y_pred = pd.Series(index=fh_abs)
        pred_int = pd.DataFrame(index=fh_abs, columns=["lower", "upper"])

        # for in-sample predictions, pmdarima requires zero-based integer indicies
        start, end = fh.to_absolute_int(self._y.index[0], self.cutoff)[[0, -1]]
        if start < 0:
            # Can't forecasts earlier to train starting point
            raise ValueError("Can't make predictions earlier to train starting point")
        elif start < diff_order:
            # Can't forecasts earlier to arima's differencing order
            # But we return NaN for these supposedly forecastable points
            start = diff_order
            if end < start:
                # since we might have forced `start` to surpass `end`
                end = diff_order
            # get rid of unforcastable points
            fh_abs = fh_abs[fh_idx >= diff_order]
            # reindex accordingly
            fh_idx = fh_idx[fh_idx >= diff_order] - diff_order

        result = self._forecaster.predict_in_sample(
            start=start,
            end=end,
            X=X,
            return_conf_int=return_pred_int,
            alpha=alpha,
        )

        if return_pred_int:
            # unpack results
            y_pred.loc[fh_abs], pred_int.loc[fh_abs] = (
                result[0][fh_idx],
                result[1][fh_idx, :],
            )
            return y_pred, pred_int
        else:
            y_pred.loc[fh_abs] = result[fh_idx]
            return y_pred

    def _predict_fixed_cutoff(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        # make prediction
        n_periods = int(fh.to_relative(self.cutoff)[-1])
        result = self._forecaster.predict(
            n_periods=n_periods,
            X=X,
            return_conf_int=return_pred_int,
            alpha=alpha,
        )

        fh_abs = fh.to_absolute(self.cutoff)
        fh_idx = fh.to_indexer(self.cutoff)
        if return_pred_int:
            y_pred, pred_int = result
            y_pred = pd.Series(y_pred[fh_idx], index=fh_abs)
            pred_int = pd.DataFrame(
                pred_int[fh_idx, :], index=fh_abs, columns=["lower", "upper"]
            )
            return y_pred, pred_int
        else:
            return pd.Series(result[fh_idx], index=fh_abs)

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        self.check_is_fitted()
        names = self._get_fitted_param_names()
        params = self._get_fitted_params()
        fitted_params = {name: param for name, param in zip(names, params)}

        if hasattr(self._forecaster, "model_"):  # AutoARIMA
            fitted_params["order"] = self._forecaster.model_.order
            fitted_params["seasonal_order"] = self._forecaster.model_.seasonal_order
            res = self._forecaster.model_.arima_res_
        elif hasattr(self._forecaster, "arima_res_"):  # ARIMA
            res = self._forecaster.arima_res_
        else:
            res = None

        for name in ["aic", "aicc", "bic", "hqic"]:
            fitted_params[name] = getattr(res, name, None)

        return fitted_params

    def _get_fitted_params(self):
        # Return parameter values under `arima_res_`
        if hasattr(self._forecaster, "model_"):  # AutoARIMA
            return self._forecaster.model_.arima_res_._results.params
        elif hasattr(self._forecaster, "arima_res_"):  # ARIMA
            return self._forecaster.arima_res_._results.params
        else:
            raise NotImplementedError()

    def _get_fitted_param_names(self):
        # Return parameter names under `arima_res_`
        if hasattr(self._forecaster, "model_"):  # AutoARIMA
            return self._forecaster.model_.arima_res_._results.param_names
        elif hasattr(self._forecaster, "arima_res_"):  # ARIMA
            return self._forecaster.arima_res_._results.param_names
        else:
            raise NotImplementedError()

    def summary(self):
        """Summary of the fitted model."""
        return self._forecaster.summary()
