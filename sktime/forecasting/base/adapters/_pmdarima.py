# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for pmdarima forecasters to be used in sktime framework."""

__author__ = ["mloning", "hyang1996", "kejsitake", "fkiraly"]
__all__ = ["_PmdArimaAdapter"]

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA


class _PmdArimaAdapter(BaseForecaster):
    """Base class for interfacing pmdarima."""

    _tags = {
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "python_dependencies": "pmdarima",
    }

    def __init__(self):
        self._forecaster = None
        super(_PmdArimaAdapter, self).__init__()

    def _instantiate_model(self):
        raise NotImplementedError("abstract method")

    def _fit(self, y, X=None, fh=None):
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
        if X is not None:
            X = X.loc[y.index]
        self._forecaster = self._instantiate_model()
        self._forecaster.fit(y, X=X)
        return self

    def _update(self, y, X=None, update_params=True):
        """Update model with data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        if update_params:
            if X is not None:
                X = X.loc[y.index]
            self._forecaster.update(y, X=X)
        return self

    def _predict(self, fh, X=None):
        """Make forecasts.

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        """
        # distinguish between in-sample and out-of-sample prediction
        fh_oos = fh.to_out_of_sample(self.cutoff)
        fh_ins = fh.to_in_sample(self.cutoff)

        # all values are out-of-sample
        if fh.is_all_out_of_sample(self.cutoff):
            return self._predict_fixed_cutoff(fh_oos, X=X)

        # all values are in-sample
        elif fh.is_all_in_sample(self.cutoff):
            return self._predict_in_sample(fh_ins, X=X)

        # both in-sample and out-of-sample values
        else:
            y_ins = self._predict_in_sample(fh_ins, X=X)
            y_oos = self._predict_fixed_cutoff(fh_oos, X=X)
            return pd.concat([y_ins, y_oos])

    def _predict_in_sample(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Generate in sample predictions.

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        """
        if hasattr(self, "order"):
            diff_order = self.order[1]
        else:
            diff_order = self._forecaster.model_.order[1]

        # Initialize return objects
        fh_abs = fh.to_absolute(self.cutoff).to_numpy()
        fh_idx = fh.to_indexer(self.cutoff, from_cutoff=False)
        y_pred = pd.Series(index=fh_abs, dtype="float64")

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
            return_conf_int=False,
            alpha=DEFAULT_ALPHA,
        )

        if return_pred_int:
            pred_ints = []
            for a in alpha:
                pred_int = pd.DataFrame(index=fh_abs, columns=["lower", "upper"])
                result = self._forecaster.predict_in_sample(
                    start=start,
                    end=end,
                    X=X,
                    return_conf_int=return_pred_int,
                    alpha=a,
                )
                pred_int.loc[fh_abs] = result[1][fh_idx, :]
                pred_ints.append(pred_int)
            # unpack results
            result = pd.Series(result[0]).iloc[fh_idx]
            y_pred.loc[fh_abs] = result
            return y_pred, pred_ints
        else:
            result = pd.Series(result).iloc[fh_idx]
            y_pred.loc[fh_abs] = result
            return y_pred

    def _predict_fixed_cutoff(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Make predictions out of sample.

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).

        Returns
        -------
        y_pred : pandas.Series
        Returns series of predicted values.
        """
        n_periods = int(fh.to_relative(self.cutoff)[-1])
        result = self._forecaster.predict(
            n_periods=n_periods,
            X=X,
            return_conf_int=False,
            alpha=DEFAULT_ALPHA,
        )

        fh_abs = fh.to_absolute(self.cutoff)
        fh_idx = fh.to_indexer(self.cutoff)
        if return_pred_int:
            pred_ints = []
            for a in alpha:
                result = self._forecaster.predict(
                    n_periods=n_periods,
                    X=X,
                    return_conf_int=True,
                    alpha=a,
                )
                pred_int = result[1]
                pred_int = pd.DataFrame(
                    pred_int[fh_idx, :], index=fh_abs, columns=["lower", "upper"]
                )
                pred_ints.append(pred_int)
            return result[0], pred_ints
        else:
            result = pd.Series(result).iloc[fh_idx]
            result.index = fh_abs
            return result

    def _predict_interval(self, fh, X=None, coverage=0.90):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        # initializaing cutoff and fh related info
        cutoff = self.cutoff
        fh_oos = fh.to_out_of_sample(cutoff)
        fh_ins = fh.to_in_sample(cutoff)
        fh_is_in_sample = fh.is_all_in_sample(cutoff)
        fh_is_oosample = fh.is_all_out_of_sample(cutoff)

        # prepare the return DataFrame - empty with correct cols
        var_names = ["Coverage"]
        int_idx = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        pred_int = pd.DataFrame(columns=int_idx)

        alpha = [1 - x for x in coverage]
        kwargs = {"X": X, "return_pred_int": True, "alpha": alpha}
        # all values are out-of-sample
        if fh_is_oosample:
            _, y_pred_int = self._predict_fixed_cutoff(fh_oos, **kwargs)

        # all values are in-sample
        elif fh_is_in_sample:
            _, y_pred_int = self._predict_in_sample(fh_ins, **kwargs)

        # if all in-sample/out-of-sample, we put y_pred_int in the required format
        if fh_is_in_sample or fh_is_oosample:
            # needs to be replaced, also seems duplicative, identical to part A
            for intervals, a in zip(y_pred_int, coverage):
                pred_int[("Coverage", a, "lower")] = intervals["lower"]
                pred_int[("Coverage", a, "upper")] = intervals["upper"]
            return pred_int

        # both in-sample and out-of-sample values (we reach this line only then)
        # in this case, we additionally need to concat in and out-of-sample returns
        _, y_ins_pred_int = self._predict_in_sample(fh_ins, **kwargs)
        _, y_oos_pred_int = self._predict_fixed_cutoff(fh_oos, **kwargs)
        for ins_int, oos_int, a in zip(y_ins_pred_int, y_oos_pred_int, coverage):
            pred_int[("Coverage", a, "lower")] = pd.concat([ins_int, oos_int])["lower"]
            pred_int[("Coverage", a, "upper")] = pd.concat([ins_int, oos_int])["upper"]

        return pred_int

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
