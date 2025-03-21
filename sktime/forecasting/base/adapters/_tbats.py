# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for using tbats forecasters in sktime framework."""

__author__ = ["mloning", "aiwalter", "k1m190r", "fkiraly"]
__all__ = ["_TbatsAdapter"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.forecasting import check_sp


class _TbatsAdapter(BaseForecaster):
    """Base class for interfacing tbats forecasting algorithms."""

    _tags = {
        "authors": ["cotterpl", "mloning", "aiwalter", "k1m190r", "fkiraly"],
        # cotterpl for tbats package
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "python_dependencies": ["tbats", "numpy<2"],
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
        self._yname = None  # .fit(y) -> y.name

        super().__init__()

    def _create_model_class(self):
        """Instantiate (T)BATS model.

        This method should write a (T)BATS model to self._ModelClass,     and should be
        overridden by concrete classes.
        """
        raise NotImplementedError

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

    def _fit(self, y, X, fh):
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
        self._create_model_class()
        self._forecaster = self._instantiate_model()
        self._forecaster = self._forecaster.fit(y)
        self._yname = y.name

        return self

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        Derived from example provided by core devs in TBATS repository
        https://github.com/intive-DataScience/tbats/blob/master/examples/

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (ignored)
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        if update_params:
            # update model state and refit parameters
            # _fit re-runs model instantiation which triggers refit
            self._fit(y=self._y, X=None, fh=self._fh)

        else:
            # update model state without refitting parameters
            # out-of-box fit tbats method will not refit parameters
            self._forecaster.fit(y=self._y)

        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : (default=None)
            NOT USED BY TBATS

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        return self._tbats_forecast(fh)

    def _get_y_pred(self, y_in_sample, y_out_sample):
        """Combine in- & out-sample prediction, slices given fh.

        Parameters
        ----------
        y_in_sample : pd.Series
            In-sample prediction
        y_out_sample : pd.Series
            Out-sample prediction

        Returns
        -------
        pd.Series
            y_pred, sliced by fh
        """
        y_pred = pd.concat([y_in_sample, y_out_sample], ignore_index=True).rename(
            "y_pred"
        )
        y_pred = pd.DataFrame(y_pred)
        # Workaround for slicing with negative index
        y_pred["idx"] = [x for x in range(-len(y_in_sample), len(y_out_sample))]
        y_pred = y_pred.loc[y_pred["idx"].isin(self.fh.to_indexer(self.cutoff).values)]
        y_pred.index = self.fh.to_absolute_index(self.cutoff)
        y_pred = y_pred["y_pred"].rename(None)
        return y_pred

    def _tbats_forecast(self, fh):
        """TBATS point forecast adapter function.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon

        Returns
        -------
        y_pred : pd.Series
            Prediction
        """
        fh = fh.to_relative(cutoff=self.cutoff)

        if not fh.is_all_in_sample(cutoff=self.cutoff):
            fh_out = fh.to_out_of_sample(cutoff=self.cutoff)
            steps = fh_out.to_pandas()[-1]
            y_out = self._forecaster.forecast(steps=steps, confidence_level=None)
        else:
            y_out = nans(len(fh))

        # y_pred combine in and out samples
        y_in_sample = pd.Series(self._forecaster.y_hat)
        y_out_sample = pd.Series(y_out)
        y_pred = self._get_y_pred(y_in_sample=y_in_sample, y_out_sample=y_out_sample)
        y_pred.name = self._yname

        return y_pred

    def _tbats_forecast_interval(self, fh, conf_lev):
        """TBATS prediction interval forecast adapter function.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        conf_lev : float
            confidence_level for TBATS

        Returns
        -------
        y_pred : pd.Series
            Prediction
        y_pred_int : pd.DataFrame
            Prediction intervals
        """
        fh = fh.to_relative(cutoff=self.cutoff)
        fh_out = fh.to_out_of_sample(cutoff=self.cutoff)

        if not fh.is_all_in_sample(cutoff=self.cutoff):
            steps = fh_out.to_pandas()[-1]
            _, tbats_ci = self._forecaster.forecast(
                steps=steps, confidence_level=conf_lev
            )
            out = pd.DataFrame(tbats_ci)
            # pred_int
            lower = pd.Series(out["lower_bound"])
            upper = pd.Series(out["upper_bound"])
            pred_int_oos = pd.DataFrame({"lower": lower, "upper": upper})
            pred_int_oos = pred_int_oos.iloc[fh_out.to_indexer()]
            pred_int_oos.index = fh_out.to_absolute_index(self.cutoff)
        else:
            pred_int_oos = pd.DataFrame(columns=["lower", "upper"])

        full_ix = fh.to_absolute_index(self.cutoff)
        pred_int = pred_int_oos.reindex(full_ix)

        return pred_int

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        Note: In-sample forecasts are set to NaNs, since TBATS does not support it.

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
            Ignored, passed for interface compatibility
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
        cutoff = self.cutoff

        # accumulator of results
        var_names = self._get_varnames()
        var_name = var_names[0]

        int_idx = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        pred_int = pd.DataFrame(columns=int_idx, index=fh.to_absolute_index(cutoff))

        for c in coverage:
            # separate treatment for "0" coverage: upper/lower = point prediction
            if c == 0:
                pred_int[(var_name, 0, "lower")] = self._tbats_forecast(fh)
                pred_int[(var_name, 0, "upper")] = pred_int[(var_name, 0, "lower")]
                continue

            # tbats prediction intervals
            tbats_pred_int = self._tbats_forecast_interval(fh, c)

            pred_int[(var_name, c, "lower")] = tbats_pred_int["lower"]
            pred_int[(var_name, c, "upper")] = tbats_pred_int["upper"]

        return pred_int

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        fitted_params = {}
        for name in self._get_fitted_param_names():
            fitted_params[name] = getattr(self._forecaster, name, None)
        return fitted_params

    def _get_fitted_param_names(self):
        """Get names of fitted parameters."""
        return self._fitted_param_names


def nans(length):
    """Return a vector of NaNs, of length `length`."""
    return np.full(length, np.nan)
